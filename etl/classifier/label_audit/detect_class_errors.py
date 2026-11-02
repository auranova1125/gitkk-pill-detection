"""기준 분류기 embedding으로 클래스 라벨 오류 후보를 만든다."""

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from classifier.checkpoint import hash_class_map, load_checkpoint
from classifier.model import build_classifier_model
from data_loading.classifier import PillCropDataset
from data_loading.classifier_package import read_samples
from data_loading.image_transforms import build_classifier_validation_transform
from etl.classifier.label_audit.review_records import preserve_human_reviews
from etl.classifier.paths import (
    DATA,
    EXPORT,
    LABEL_AUDIT_OUTPUT,
    LABEL_AUDIT_RECORDS,
    save_table,
)
from utils.runtime import select_device


def _stack_images(batch: list[dict[str, Any]]) -> torch.Tensor:
    return torch.stack([sample["image"] for sample in batch])


def extract_embeddings(
    samples: pd.DataFrame,
    *,
    package_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """모든 crop의 avgpool embedding과 softmax 확률을 계산한다."""

    class_map = json.loads((package_dir / "class_map.json").read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    if checkpoint["class_map_hash"] != hash_class_map(class_map):
        raise ValueError("checkpoint class map differs from the package class map")
    model = build_classifier_model(class_map["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)

    captured: dict[str, torch.Tensor] = {}

    def capture_embedding(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["embedding"] = output.flatten(1)

    hook = model.avgpool.register_forward_hook(capture_embedding)
    loader = DataLoader(
        PillCropDataset(
            samples,
            package_dir,
            build_classifier_validation_transform(),
        ),
        batch_size=batch_size,
        num_workers=2 if device.type == "cuda" else 0,
        collate_fn=_stack_images,
    )
    embeddings: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for images in loader:
                logits = model(images.to(device))
                embeddings.append(captured["embedding"].cpu().numpy())
                probabilities.append(logits.softmax(1).cpu().numpy())
    finally:
        hook.remove()
    return (
        np.concatenate(embeddings).astype(np.float32),
        np.concatenate(probabilities).astype(np.float32),
    )


def compare_group_embeddings(
    samples: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """같은 조합의 다른 사진에서 라벨·타 클래스 평균 유사도를 비교한다."""

    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    own_similarity = np.full(len(samples), np.nan)
    other_similarity = np.full(len(samples), np.nan)
    other_class = np.full(len(samples), -1, dtype=int)
    for indices in samples.groupby("group_id").indices.values():
        similarity = normalized[indices] @ normalized[indices].T
        image_ids = samples["image_id"].to_numpy()[indices]
        classes = samples["class_index"].to_numpy()[indices]
        for local_index in range(len(indices)):
            other_image = image_ids != image_ids[local_index]
            if not other_image.any():
                continue
            by_class = (
                pd.Series(similarity[local_index, other_image])
                .groupby(classes[other_image])
                .mean()
            )
            class_index = classes[local_index]
            if class_index in by_class.index:
                own_similarity[indices[local_index]] = by_class[class_index]
            alternatives = by_class.drop(class_index, errors="ignore")
            if not alternatives.empty:
                other_similarity[indices[local_index]] = alternatives.max()
                other_class[indices[local_index]] = int(alternatives.idxmax())
    return own_similarity, other_similarity, other_class


def cross_validate_embeddings(
    samples: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    num_classes: int,
    folds: int = 5,
) -> np.ndarray:
    """그룹 단위 교차 검증으로 각 crop의 out-of-fold 확률을 만든다."""

    group_count = samples["group_id"].nunique()
    if folds < 2 or folds > group_count:
        raise ValueError("folds must be between 2 and the number of groups")
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    targets = samples["class_index"].to_numpy()
    probabilities = np.zeros((len(samples), num_classes), dtype=np.float32)
    splitter = GroupKFold(folds)
    for train_indices, test_indices in splitter.split(
        normalized,
        targets,
        samples["group_id"].to_numpy(),
    ):
        classifier = LogisticRegression(C=10, max_iter=2000)
        classifier.fit(normalized[train_indices], targets[train_indices])
        probabilities[np.ix_(test_indices, classifier.classes_)] = (
            classifier.predict_proba(normalized[test_indices])
        )
    return probabilities


def find_label_swaps(samples: pd.DataFrame, scores: pd.DataFrame) -> np.ndarray:
    """같은 사진의 두 crop이 서로 상대 클래스로 판단되는지 확인한다."""

    swaps = np.zeros(len(samples), dtype=bool)
    for indices in samples.groupby("image_id").indices.values():
        for first_position, first_index in enumerate(indices):
            for second_index in indices[first_position + 1 :]:
                if (
                    scores.loc[first_index, "suspect_class"]
                    == samples.loc[second_index, "class_index"]
                    and scores.loc[second_index, "suspect_class"]
                    == samples.loc[first_index, "class_index"]
                ):
                    swaps[[first_index, second_index]] = True
    return swaps


def build_class_scores(
    samples: pd.DataFrame,
    embeddings: np.ndarray,
    model_probabilities: np.ndarray,
    oof_probabilities: np.ndarray,
) -> pd.DataFrame:
    """embedding 비교와 OOF 예측을 클래스 오류 후보 점수로 합친다."""

    samples = samples.reset_index(drop=True)
    if embeddings.shape[0] != len(samples):
        raise ValueError("embedding count must match samples")
    if model_probabilities.shape != oof_probabilities.shape:
        raise ValueError("model and OOF probability shapes must match")
    if model_probabilities.shape[0] != len(samples):
        raise ValueError("probability count must match samples")

    own_similarity, other_similarity, other_class = compare_group_embeddings(
        samples,
        embeddings,
    )
    targets = samples["class_index"].to_numpy()
    row_indices = np.arange(len(samples))
    code_by_class = (
        samples.drop_duplicates("class_index")
        .set_index("class_index")["code"]
        .to_dict()
    )
    scores = samples[
        [
            "sample_id",
            "image_id",
            "group_id",
            "code",
            "name",
            "split",
            "source",
            "class_index",
            "crop_path",
        ]
    ].copy()
    scores["model_pred"] = model_probabilities.argmax(axis=1)
    scores["model_label_prob"] = model_probabilities[row_indices, targets].round(4)
    scores["oof_pred"] = oof_probabilities.argmax(axis=1)
    scores["oof_label_prob"] = oof_probabilities[row_indices, targets].round(4)
    scores["group_own_sim"] = np.round(own_similarity, 4)
    scores["group_other_sim"] = np.round(other_similarity, 4)
    scores["group_other_class"] = other_class
    scores["group_gap"] = np.round(other_similarity - own_similarity, 4)
    group_flag = scores["group_gap"] > 0
    oof_flag = scores["oof_pred"] != targets
    scores["suspect_class"] = np.where(
        group_flag,
        scores["group_other_class"],
        scores["oof_pred"],
    )
    scores["suspect_code"] = scores["suspect_class"].map(code_by_class)
    scores["swap"] = find_label_swaps(samples, scores)
    scores["priority"] = np.select(
        [scores["swap"] | (group_flag & oof_flag), group_flag | oof_flag],
        [1, 2],
        default=0,
    )
    return scores


def _known_errors(records_dir: Path) -> tuple[set[str], set[str]]:
    sample_ids: set[str] = set()
    image_ids: set[str] = set()
    for file_name in ("position_suspects.csv", "class_suspects.csv"):
        path = records_dir / file_name
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        errors = frame[frame["판정"] == "오류"]
        sample_ids.update(errors["sample_id"])
        image_ids.update(errors.loc[errors["조치"] == "이미지 제외", "image_id"])
    return sample_ids, image_ids


def _crop_tile(
    path: Path,
    *,
    size: int,
    text: str,
    color: tuple[int, int, int],
    known_error: bool = False,
) -> np.ndarray:
    with Image.open(path) as source_image:
        image = np.array(source_image.convert("RGB"))
    height, width = image.shape[:2]
    scale = size / max(height, width)
    image = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
    )
    tile = np.full((size + 22, size, 3), 255, dtype=np.uint8)
    tile[: image.shape[0], : image.shape[1]] = image
    cv2.putText(tile, text, (2, size + 16), 0, 0.42, color, 1)
    if known_error:
        tile[:size] = (tile[:size] * 0.4 + 255 * 0.6).astype(np.uint8)
        cv2.line(tile, (0, 0), (size, size), (120, 120, 120), 3)
        cv2.line(tile, (size, 0), (0, size), (120, 120, 120), 3)
        cv2.putText(tile, "known error", (4, 20), 0, 0.5, (90, 90, 90), 1)
    return tile


def _draw_review_sheet(
    rows: list[tuple[int, Any]],
    samples: pd.DataFrame,
    *,
    package_dir: Path,
    records_dir: Path,
    path: Path,
    size: int = 150,
) -> None:
    bad_samples, bad_images = _known_errors(records_dir)

    def is_known_error(sample: Any) -> bool:
        return sample.sample_id in bad_samples or sample.image_id in bad_images

    blank = np.full((size + 22, size, 3), 255, dtype=np.uint8)
    lines: list[np.ndarray] = []
    for number, row in rows:
        group = samples[
            (samples["group_id"] == row.group_id)
            & (samples["image_id"] != row.image_id)
        ]
        same_class = list(
            group[group["class_index"] == row.class_index].head(2).itertuples()
        )
        suspect_class = list(
            group[group["class_index"] == row.suspect_class].head(2).itertuples()
        )
        label = blank.copy()
        cv2.putText(label, str(number), (10, 60), 0, 1.6, (0, 0, 0), 3)
        cv2.putText(label, row.code[-6:], (10, 100), 0, 0.6, (0, 0, 0), 1)
        cv2.putText(
            label,
            f"-> {row.suspect_code[-6:]}",
            (10, 125),
            0,
            0.6,
            (200, 0, 0),
            1,
        )
        tiles = [
            label,
            _crop_tile(
                package_dir / row.crop_path,
                size=size,
                text="suspect",
                color=(200, 0, 0),
            ),
        ]
        tiles.extend(
            _crop_tile(
                package_dir / sample.crop_path,
                size=size,
                text=f"label {row.code[-6:]}",
                color=(0, 140, 0),
                known_error=is_known_error(sample),
            )
            for sample in same_class
        )
        tiles.extend([blank] * (2 - len(same_class)))
        tiles.extend(
            _crop_tile(
                package_dir / sample.crop_path,
                size=size,
                text=f"other {row.suspect_code[-6:]}",
                color=(200, 0, 0),
                known_error=is_known_error(sample),
            )
            for sample in suspect_class
        )
        tiles.extend([blank] * (2 - len(suspect_class)))
        line = np.hstack(tiles)
        line[-1:, :] = 180
        lines.append(line)
    sheet = np.vstack(lines)
    if not cv2.imwrite(
        str(path),
        cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    ):
        raise OSError(f"failed to write review sheet: {path}")


def detect_class_errors(
    *,
    checkpoint_path: Path,
    package_dir: Path = EXPORT / "training_package_classifier",
    records_dir: Path = LABEL_AUDIT_RECORDS,
    output_dir: Path = LABEL_AUDIT_OUTPUT,
    device: torch.device | None = None,
    recompute: bool = False,
    folds: int = 5,
) -> pd.DataFrame:
    """embedding과 OOF 예측으로 클래스 오류 후보를 만들고 검수표를 갱신한다."""

    selected_device = device or select_device()
    samples = read_samples(package_dir).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "class_embeddings.npz"
    if cache_path.exists() and not recompute:
        cache = np.load(cache_path)
        if list(cache["sample_id"]) != samples["sample_id"].tolist():
            raise ValueError("cached embedding sample order differs; use recompute")
        embeddings = cache["emb"]
        model_probabilities = cache["prob"]
    else:
        embeddings, model_probabilities = extract_embeddings(
            samples,
            package_dir=package_dir,
            checkpoint_path=checkpoint_path,
            device=selected_device,
        )
        np.savez_compressed(
            cache_path,
            emb=embeddings,
            prob=model_probabilities,
            sample_id=samples["sample_id"].to_numpy(dtype=str),
        )

    oof_probabilities = cross_validate_embeddings(
        samples,
        embeddings,
        num_classes=model_probabilities.shape[1],
        folds=folds,
    )
    scores = build_class_scores(
        samples,
        embeddings,
        model_probabilities,
        oof_probabilities,
    )
    save_table(scores, output_dir / "class_scores.csv")
    suspects = scores[scores["priority"] > 0].sort_values(
        ["priority", "oof_label_prob"]
    )
    suspects = suspects.copy()
    suspects["그림번호"] = suspects.groupby("priority").cumcount()
    suspects["그림파일"] = [
        f"review/c{priority}_review_{number // 20 + 1}.jpg"
        for priority, number in zip(
            suspects["priority"],
            suspects["그림번호"],
            strict=True,
        )
    ]
    position_path = records_dir / "position_suspects.csv"
    if position_path.exists():
        position = pd.read_csv(position_path, dtype=str, keep_default_na=False)
        position_errors = position[position["판정"] == "오류"]
    else:
        position_errors = pd.DataFrame(columns=["sample_id", "image_id", "조치"])
    bad_samples = set(position_errors["sample_id"])
    bad_images = set(
        position_errors.loc[position_errors["조치"] == "이미지 제외", "image_id"]
    )
    suspects["위치검수와중복"] = suspects["sample_id"].isin(bad_samples) | suspects[
        "image_id"
    ].isin(bad_images)
    review_path = records_dir / "class_suspects.csv"
    suspects = preserve_human_reviews(suspects, review_path)
    leading_columns = ["그림번호", "그림파일"]
    ordered_columns = leading_columns + [
        column for column in suspects.columns if column not in leading_columns
    ]
    save_table(suspects[ordered_columns], review_path)

    review_dir = output_dir / "review"
    review_dir.mkdir(exist_ok=True)
    for old_path in review_dir.glob("c*_review_*.jpg"):
        old_path.unlink()
    for priority in (1, 2):
        group = suspects[suspects["priority"] == priority]
        for start in range(0, len(group), 20):
            _draw_review_sheet(
                list(
                    zip(
                        group["그림번호"].iloc[start : start + 20],
                        group.iloc[start : start + 20].itertuples(),
                        strict=True,
                    )
                ),
                samples,
                package_dir=package_dir,
                records_dir=records_dir,
                path=review_dir / f"c{priority}_review_{start // 20 + 1}.jpg",
            )
    return suspects


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DATA / "best.pt")
    parser.add_argument(
        "--package-dir", type=Path, default=EXPORT / "training_package_classifier"
    )
    parser.add_argument("--records-dir", type=Path, default=LABEL_AUDIT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=LABEL_AUDIT_OUTPUT)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    suspects = detect_class_errors(
        checkpoint_path=args.checkpoint,
        package_dir=args.package_dir,
        records_dir=args.records_dir,
        output_dir=args.output_dir,
        recompute=args.recompute,
        folds=args.folds,
    )
    print(f"클래스 오류 후보 crop {len(suspects)}개")


if __name__ == "__main__":
    main()
