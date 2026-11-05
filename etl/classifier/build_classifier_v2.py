"""라벨 2차 검수 조치를 v1 분류 패키지에 적용해 v2를 만든다."""

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch

from classifier.evaluate import EvaluationResult, evaluate_with_predictions
from classifier.inference import load_classifier_inference_model
from classifier.training_data import validate_classifier_training_data
from data_loading.classifier import build_classifier_inputs
from data_loading.classifier_package import (
    calculate_classifier_split_hash,
    read_classifier_samples,
    summarize_classifier_samples,
)
from etl.classifier.paths import (
    DATA,
    EXPORT,
    LABEL_AUDIT_OUTPUT,
    LABEL_AUDIT_RECORDS,
    REPORTS,
)
from utils.io import read_json, write_json
from utils.runtime import select_device

SOURCE_SPLIT_HASH = "49d2afee86a4ec8f48c25488eac9b1f9081b36066e5533eba40670087f60430c"
DROP_ACTIONS = {"이미지 제외", "bbox 제외"}
SWAP_ACTION = "라벨 교환 (수정)"
ALLOWED_ACTIONS = DROP_ACTIONS | {SWAP_ACTION}


@dataclass(frozen=True, slots=True)
class ClassifierV2BuildResult:
    """생성된 v2 패키지 위치와 데이터 요약."""

    package_dir: Path
    zip_path: Path
    data_summary: dict[str, Any]


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    name: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def apply_label_actions(
    source_samples: pd.DataFrame,
    actions: pd.DataFrame,
    class_map: dict[str, Any],
) -> pd.DataFrame:
    """v1 samples에 제외·라벨 교환 조치를 적용한 v2 samples를 반환한다."""

    _require_columns(
        source_samples,
        {
            "sample_id",
            "image_id",
            "code",
            "class_index",
            "category_id",
            "item_seq",
            "name",
            "crop_path",
        },
        name="source_samples",
    )
    _require_columns(
        actions,
        {"sample_id", "image_id", "code", "new_code", "action"},
        name="actions",
    )
    if not source_samples["sample_id"].is_unique:
        raise ValueError("source sample_id values must be unique")
    if not actions["sample_id"].is_unique:
        raise ValueError("each sample must have at most one final action")
    unknown_actions = set(actions["action"]) - ALLOWED_ACTIONS
    if unknown_actions:
        raise ValueError(f"unknown label audit actions: {sorted(unknown_actions)}")
    missing_samples = set(actions["sample_id"]) - set(source_samples["sample_id"])
    if missing_samples:
        raise ValueError(
            "actions reference samples outside the source package: "
            f"{sorted(missing_samples)[:3]}"
        )

    source_by_id = source_samples.set_index("sample_id", drop=False)
    for action in actions.to_dict("records"):
        source = source_by_id.loc[action["sample_id"]]
        if source["image_id"] != action["image_id"] or source["code"] != action["code"]:
            raise ValueError(
                f"action metadata differs from source sample: {action['sample_id']}"
            )

    excluded_images = set(actions.loc[actions["action"] == "이미지 제외", "image_id"])
    expected_image_samples = set(
        source_samples.loc[
            source_samples["image_id"].isin(excluded_images),
            "sample_id",
        ]
    )
    recorded_image_samples = set(
        actions.loc[
            (actions["image_id"].isin(excluded_images))
            & (actions["action"] == "이미지 제외"),
            "sample_id",
        ]
    )
    if not expected_image_samples <= recorded_image_samples:
        missing = sorted(expected_image_samples - recorded_image_samples)
        raise ValueError(f"image exclusion is missing samples: {missing[:3]}")

    class_entries = class_map.get("classes")
    if not isinstance(class_entries, list):
        raise ValueError("class_map.classes must be a list")
    classes_by_code = {
        f"K-{entry['category_id']:06d}": entry for entry in class_entries
    }

    dropped_sample_ids = set(
        actions.loc[actions["action"].isin(DROP_ACTIONS), "sample_id"]
    )
    result = source_samples[
        ~source_samples["sample_id"].isin(dropped_sample_ids)
    ].copy()
    result_by_id = result.set_index("sample_id", drop=False)
    for action in actions.loc[actions["action"] == SWAP_ACTION].to_dict("records"):
        sample_id = action["sample_id"]
        new_code = action["new_code"]
        if new_code not in classes_by_code:
            raise ValueError(f"swap target is missing from class_map: {new_code}")
        entry = classes_by_code[new_code]
        result_by_id.loc[
            sample_id,
            ["code", "class_index", "category_id", "item_seq", "name"],
        ] = [
            new_code,
            str(entry["class_index"]),
            str(entry["category_id"]),
            str(entry["item_seq"]),
            entry["name"],
        ]
        result_by_id.loc[sample_id, "crop_path"] = f"crops/{new_code}/{sample_id}.png"
    return result_by_id.reset_index(drop=True)


def write_classifier_v2_package(
    *,
    source_package_dir: Path,
    output_package_dir: Path,
    source_samples: pd.DataFrame,
    v2_samples: pd.DataFrame,
    overwrite: bool = False,
) -> ClassifierV2BuildResult:
    """검증된 v2 samples와 crop을 새 학습 패키지 및 zip으로 기록한다."""

    source_package_dir = Path(source_package_dir).resolve()
    output_package_dir = Path(output_package_dir).resolve()
    zip_path = output_package_dir.with_suffix(".zip")
    if source_package_dir == output_package_dir:
        raise ValueError("source and output package directories must differ")
    existing = [path for path in (output_package_dir, zip_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"classifier v2 output already exists: {names}")
    output_package_dir.parent.mkdir(parents=True, exist_ok=True)

    class_map = read_json(source_package_dir / "class_map.json")
    old_crop_paths = source_samples.set_index("sample_id")["crop_path"]
    with tempfile.TemporaryDirectory(
        prefix="classifier-v2-",
        dir=output_package_dir.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_package = temporary_root / output_package_dir.name
        temporary_package.mkdir()
        for sample in v2_samples.to_dict("records"):
            sample_id = sample["sample_id"]
            destination = temporary_package / sample["crop_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                source_package_dir / old_crop_paths.loc[sample_id],
                destination,
            )
        v2_samples.to_csv(temporary_package / "samples.csv", index=False)
        write_json(temporary_package / "class_map.json", class_map)
        summary = summarize_classifier_samples(
            read_classifier_samples(temporary_package),
            class_map["num_classes"],
        )
        write_json(temporary_package / "data_summary.json", summary)

        config = SimpleNamespace(
            batch_size=32,
            seed=42,
            data_dir=temporary_package,
        )
        training_data = build_classifier_inputs(
            config=config,
            device=torch.device("cpu"),
        )
        validate_classifier_training_data(
            training_data,
        )
        temporary_zip = Path(
            shutil.make_archive(
                str(temporary_root / output_package_dir.name),
                "zip",
                temporary_package,
            )
        )

        if output_package_dir.exists():
            shutil.rmtree(output_package_dir)
        if zip_path.exists():
            zip_path.unlink()
        temporary_package.replace(output_package_dir)
        temporary_zip.replace(zip_path)
    return ClassifierV2BuildResult(
        package_dir=output_package_dir,
        zip_path=zip_path,
        data_summary=summary,
    )


def evaluate_package(
    checkpoint_path: Path,
    package_dir: Path,
) -> tuple[EvaluationResult, pd.DataFrame]:
    """기준 checkpoint를 지정한 분류 패키지의 validation에서 평가한다."""

    device = select_device()
    class_map = read_json(package_dir / "class_map.json")
    config = SimpleNamespace(
        batch_size=64,
        seed=42,
        data_dir=package_dir,
    )
    _, validation_loader, _, _ = build_classifier_inputs(
        config=config,
        device=device,
    )
    model, _ = load_classifier_inference_model(
        checkpoint_path,
        class_map,
        device=device,
    )
    artifacts = evaluate_with_predictions(
        model,
        validation_loader,
        class_map,
        torch.nn.CrossEntropyLoss(),
        device,
    )
    predictions = pd.DataFrame(
        [prediction.to_dict() for prediction in artifacts.predictions]
    )
    return artifacts.metrics, predictions


def build_classifier_v2(
    *,
    source_package_dir: Path = EXPORT / "training_package_classifier",
    output_package_dir: Path = EXPORT / "training_package_classifier_v2",
    actions_path: Path = LABEL_AUDIT_RECORDS / "label_actions.csv",
    expected_source_split_hash: str = SOURCE_SPLIT_HASH,
    overwrite: bool = False,
) -> tuple[ClassifierV2BuildResult, pd.DataFrame, pd.DataFrame]:
    """v1 패키지를 확인하고 확정 조치를 적용해 v2 패키지를 만든다."""

    source_package_dir = Path(source_package_dir)
    source_samples = pd.read_csv(
        source_package_dir / "samples.csv",
        dtype=str,
        keep_default_na=False,
    )
    actual_hash = calculate_classifier_split_hash(
        read_classifier_samples(source_package_dir)
    )
    if actual_hash != expected_source_split_hash:
        raise ValueError(
            f"source split hash differs from the audited package: {actual_hash}"
        )
    actions = pd.read_csv(actions_path, dtype=str, keep_default_na=False)
    class_map = read_json(source_package_dir / "class_map.json")
    v2_samples = apply_label_actions(source_samples, actions, class_map)
    result = write_classifier_v2_package(
        source_package_dir=source_package_dir,
        output_package_dir=output_package_dir,
        source_samples=source_samples,
        v2_samples=v2_samples,
        overwrite=overwrite,
    )
    return result, source_samples, actions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-package",
        type=Path,
        default=EXPORT / "training_package_classifier",
    )
    parser.add_argument(
        "--output-package",
        type=Path,
        default=EXPORT / "training_package_classifier_v2",
    )
    parser.add_argument(
        "--actions",
        type=Path,
        default=LABEL_AUDIT_RECORDS / "label_actions.csv",
    )
    parser.add_argument(
        "--expected-source-split-hash",
        default=SOURCE_SPLIT_HASH,
    )
    parser.add_argument("--checkpoint", type=Path, default=DATA / "best.pt")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result, source_samples, actions = build_classifier_v2(
        source_package_dir=args.source_package,
        output_package_dir=args.output_package,
        actions_path=args.actions,
        expected_source_split_hash=args.expected_source_split_hash,
        overwrite=args.overwrite,
    )
    print(
        f"v1 {len(source_samples)} → v2 "
        f"{result.data_summary['num_train_samples'] + result.data_summary['num_val_samples']}"
    )
    print(f"저장: {result.package_dir}, {result.zip_path}")

    if not args.no_eval and args.checkpoint.is_file():
        evaluation_dir = LABEL_AUDIT_OUTPUT.parent / "data_v2_eval"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        for name, package_dir in (
            ("v1", args.source_package),
            ("v2", result.package_dir),
        ):
            metrics, predictions = evaluate_package(args.checkpoint, package_dir)
            predictions.to_csv(
                evaluation_dir / f"baseline_{name}_val_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
            write_json(
                evaluation_dir / f"baseline_{name}_val_metrics.json",
                metrics.to_dict(),
            )
            print(
                f"기준 모델 {name} val: Top-1 {metrics.top1_accuracy:.4f}, "
                f"macro F1 {metrics.macro_f1:.4f}"
            )
    elif not args.no_eval:
        print(f"기준 모델 없음({args.checkpoint}): 평가를 건너뜁니다")

    report_path = REPORTS / "data_v2_report.md"
    if not report_path.exists():
        print(f"결과 보고서는 통합된 기준 문서를 사용합니다: {report_path}")
    print(f"적용 조치 {len(actions)}개")


if __name__ == "__main__":
    main()
