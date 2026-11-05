"""확정된 classifier 검수 조치를 기존 학습 패키지에 적용한다."""

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_loading.classifier_package import (
    read_classifier_samples,
    summarize_classifier_samples,
)
from etl.classifier.build_classifier_v2 import apply_label_actions
from utils.io import read_json, write_json

DEFAULT_ACTIONS_PATH = (
    Path(__file__).resolve().parent / "label_audit" / "records" / "label_actions.csv"
)


@dataclass(frozen=True, slots=True)
class ClassifierActionResult:
    """Classifier 검수 조치 적용 결과."""

    excluded_images: int
    excluded_samples: int
    relabeled_samples: int
    train_samples: int
    validation_samples: int


def apply_classifier_review_actions(
    package_dir: Path,
    *,
    actions_path: Path = DEFAULT_ACTIONS_PATH,
) -> ClassifierActionResult:
    """현재 samples에 확정 제외·라벨 수정 조치를 적용하고 summary를 갱신한다."""

    package_dir = Path(package_dir).expanduser().resolve()
    samples_path = package_dir / "samples.csv"
    summary_path = package_dir / "data_summary.json"
    source_samples = pd.read_csv(samples_path, dtype=str, keep_default_na=False)
    actions = pd.read_csv(actions_path, dtype=str, keep_default_na=False)
    class_map = read_json(package_dir / "class_map.json")
    updated_samples = apply_label_actions(source_samples, actions, class_map)
    original_samples = samples_path.read_bytes()
    original_summary = summary_path.read_bytes()

    swap_actions = actions[actions["action"] == "라벨 교환 (수정)"]
    source_by_id = source_samples.set_index("sample_id")
    updated_by_id = updated_samples.set_index("sample_id")
    copied_paths: list[Path] = []
    try:
        for action in swap_actions.to_dict("records"):
            sample_id = action["sample_id"]
            source_path = package_dir / source_by_id.loc[sample_id, "crop_path"]
            destination_path = package_dir / updated_by_id.loc[sample_id, "crop_path"]
            if source_path != destination_path:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                destination_existed = destination_path.exists()
                shutil.copy2(source_path, destination_path)
                if not destination_existed:
                    copied_paths.append(destination_path)

        with tempfile.TemporaryDirectory(
            prefix="classifier-actions-",
            dir=package_dir.parent,
        ) as temporary_directory:
            temporary_package = Path(temporary_directory)
            temporary_samples_path = temporary_package / "samples.csv"
            temporary_summary_path = temporary_package / "data_summary.json"
            updated_samples.to_csv(temporary_samples_path, index=False)
            runtime_samples = read_classifier_samples(temporary_package)
            summary = summarize_classifier_samples(
                runtime_samples,
                class_map["num_classes"],
            )
            write_json(temporary_summary_path, summary)
            temporary_samples_path.replace(samples_path)
            temporary_summary_path.replace(summary_path)
    except Exception:
        samples_path.write_bytes(original_samples)
        summary_path.write_bytes(original_summary)
        for path in copied_paths:
            path.unlink(missing_ok=True)
        raise

    excluded_image_count = actions.loc[
        actions["action"] == "이미지 제외",
        "image_id",
    ].nunique()
    return ClassifierActionResult(
        excluded_images=int(excluded_image_count),
        excluded_samples=len(source_samples) - len(updated_samples),
        relabeled_samples=len(swap_actions),
        train_samples=int((updated_samples["split"] == "train").sum()),
        validation_samples=int((updated_samples["split"] == "val").sum()),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS_PATH)
    args = parser.parse_args(argv)
    result = apply_classifier_review_actions(
        args.package_dir,
        actions_path=args.actions,
    )
    print(
        f"조치 완료: 제외 이미지 {result.excluded_images}, "
        f"제외 crop {result.excluded_samples}, "
        f"라벨 수정 {result.relabeled_samples}, "
        f"train {result.train_samples}, validation {result.validation_samples}"
    )


if __name__ == "__main__":
    main()
