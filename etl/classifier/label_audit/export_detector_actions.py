"""분류 라벨 검수 조치를 탐지 데이터의 이미지 기준 조치로 변환한다."""

import argparse
import io
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from etl.classifier.paths import (
    DETECTOR_PACKAGE,
    LABEL_AUDIT_RECORDS,
    save_table,
)

ACTION_PRIORITY = ["이미지 제외", "bbox 제외", "라벨 교환 (수정)"]
DETECTOR_RECOMMENDATION = {
    "이미지 제외": "이미지 제외",
    "bbox 제외": "bbox 수정 또는 이미지 제외",
    "라벨 교환 (수정)": "drug_code 메타데이터 수정 (탐지 label은 pill 유지)",
}


def load_detector_manifest(package_path: Path) -> pd.DataFrame:
    """압축 또는 디렉터리 형태 탐지 패키지에서 split manifest를 읽는다."""

    package_path = Path(package_path)
    if package_path.is_dir():
        return pd.read_csv(package_path / "split_manifest.csv")
    with zipfile.ZipFile(package_path) as archive:
        return pd.read_csv(io.BytesIO(archive.read("split_manifest.csv")))


def build_detector_actions(
    actions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """분류 crop 조치를 탐지 manifest에 연결한 이미지별 조치로 변환한다."""

    required_actions = {
        "sample_id",
        "image_id",
        "code",
        "new_code",
        "action",
        "split",
        "source",
        "출처",
        "메모",
    }
    missing_actions = required_actions - set(actions.columns)
    if missing_actions:
        raise ValueError(f"actions are missing columns: {sorted(missing_actions)}")
    required_manifest = {"image_path", "dataset", "split", "group_id"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(
            f"detector manifest is missing columns: {sorted(missing_manifest)}"
        )
    unknown_actions = set(actions["action"]) - set(ACTION_PRIORITY)
    if unknown_actions:
        raise ValueError(f"unknown actions: {sorted(unknown_actions)}")

    manifest = manifest.copy()
    manifest["file_name"] = manifest["image_path"].map(
        lambda value: PurePosixPath(str(value).replace("\\", "/")).name
    )
    manifest["source_image_id"] = (
        manifest["file_name"]
        .str.replace(r"^(source_\d+|codeit)_", "", regex=True)
        .str.removesuffix(".png")
    )

    ranked = actions.copy()
    ranked["priority"] = ranked["action"].map(ACTION_PRIORITY.index)
    per_image = (
        ranked.sort_values("priority")
        .groupby("image_id", as_index=False)
        .agg(
            분류기_조치=("action", "first"),
            source=("source", "first"),
            분류기_split=("split", "first"),
            출처=("출처", lambda values: ", ".join(sorted(set(values)))),
            메모=("메모", lambda values: " / ".join(dict.fromkeys(values))),
        )
    )
    per_image["탐지기_권장_조치"] = per_image["분류기_조치"].map(
        DETECTOR_RECOMMENDATION
    )
    partial = (
        actions[actions["action"] != "이미지 제외"]
        .groupby("image_id")["code"]
        .agg(lambda values: ", ".join(values))
    )
    per_image["대상_알약"] = per_image["image_id"].map(partial).fillna("전체")
    swaps = actions[actions["action"] == "라벨 교환 (수정)"]
    for image_id, group in swaps.groupby("image_id"):
        mappings = ", ".join(
            f"{code} → {new_code}"
            for code, new_code in zip(
                group["code"],
                group["new_code"],
                strict=True,
            )
        )
        per_image.loc[per_image["image_id"] == image_id, "메모"] = mappings

    merged = per_image.merge(
        manifest[
            [
                "source_image_id",
                "file_name",
                "image_path",
                "dataset",
                "split",
                "group_id",
            ]
        ],
        left_on="image_id",
        right_on="source_image_id",
        how="left",
    )
    missing_image_ids = tuple(
        sorted(merged.loc[merged["file_name"].isna(), "image_id"])
    )
    merged = merged[merged["file_name"].notna()].rename(
        columns={"split": "탐지기_split", "image_id": "원본_사진"}
    )
    output_columns = [
        "file_name",
        "image_path",
        "dataset",
        "탐지기_split",
        "group_id",
        "탐지기_권장_조치",
        "대상_알약",
        "분류기_조치",
        "출처",
        "메모",
        "원본_사진",
    ]
    output = merged[output_columns].sort_values(["탐지기_split", "file_name"])
    return output.reset_index(drop=True), missing_image_ids


def export_detector_actions(
    *,
    package_path: Path = DETECTOR_PACKAGE,
    actions_path: Path = LABEL_AUDIT_RECORDS / "label_actions.csv",
    output_path: Path = LABEL_AUDIT_RECORDS / "detector_actions.csv",
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """확정 분류 조치와 탐지 manifest를 읽어 탐지 조치 CSV를 갱신한다."""

    actions = pd.read_csv(actions_path, dtype=str, keep_default_na=False)
    manifest = load_detector_manifest(package_path)
    output, missing_image_ids = build_detector_actions(actions, manifest)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_table(output, output_path)
    return output, missing_image_ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DETECTOR_PACKAGE)
    parser.add_argument(
        "--actions",
        type=Path,
        default=LABEL_AUDIT_RECORDS / "label_actions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LABEL_AUDIT_RECORDS / "detector_actions.csv",
    )
    args = parser.parse_args(argv)
    output, missing_image_ids = export_detector_actions(
        package_path=args.package,
        actions_path=args.actions,
        output_path=args.output,
    )
    print(f"탐지 데이터에 있는 오류 사진 {len(output)}장")
    if missing_image_ids:
        print(f"탐지 데이터에 없는 사진 {len(missing_image_ids)}장")


if __name__ == "__main__":
    main()
