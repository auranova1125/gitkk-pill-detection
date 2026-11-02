"""AI Hub val(조합)로 분류기 외부 평가 패키지를 만든다. 학습에는 쓰지 않는다.

순서
    1. 1차 라벨 검수: 학습 데이터와 같은 validate_source_labels 규칙 (이미지째 제외)
    2. 촬영 조건·클래스: 연회색 배경, 주백색, 70/75/90°, 118종 class_map 안의 약만
    3. 학습 데이터와 중복: 픽셀 MD5가 같은 사진은 제외, pHash 거리 ≤ 5인 사진은 near_dup_in_train=True로 표시
       (학습 패키지 val과 같은 규칙. 평가 때 전체 점수와 표시를 뺀 점수를 함께 본다)
    4. 2차 라벨 검수 반영: records/의 사람 판정(위치·클래스)이 있으면 적용
    5. crop: 학습과 같은 cut_pill(여백 5% + clip), class_map은 학습 패키지 것을 그대로

결과
    data/classifier/export/eval_package_aihub_val/(.zip)
        samples.csv, class_map.json, eval_summary.json, excluded.csv, crops/<약 코드>/<sample_id>.png
    data/classifier/aihub_val/   1차 검수 목록, 학습 사진 hash, 중복 목록
    reports/data/classifier/aihub_val_validation_report.md   1차 라벨 검수
    reports/data/classifier/aihub_val_eval_package_report.md 평가 패키지 구성

2차 검수 (패키지를 만든 뒤)
    python -m etl.classifier.label_audit.detect_bbox_errors --package-dir <패키지> \\
        --records-dir etl/classifier/aihub_val/records --output-dir data/classifier/aihub_val/label_audit \\
        --image-zip <VS_1_조합.zip>
    python -m etl.classifier.label_audit.detect_class_errors --package-dir <패키지> \\
        --records-dir etl/classifier/aihub_val/records --output-dir data/classifier/aihub_val/label_audit
    → 사람이 판정을 채운 뒤 이 스크립트를 --overwrite로 다시 실행하면 판정이 반영된다

사용:
    python -m etl.classifier.aihub_val.build_eval_package [--overwrite]
"""

import argparse
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from data_loading.classifier import (
    EVAL_SPLIT,
    build_classifier_eval_loader,
    make_eval_summary,
    read_samples,
)
from data_loading.image_transforms import CROP_MARGIN, cut_pill
from etl.classifier.build_classifier_v2 import apply_label_actions
from etl.classifier.label_audit.merge_review_actions import build_label_actions
from etl.classifier.paths import (
    AIHUB_VAL_META,
    AIHUB_VAL_PACKAGE,
    AIHUB_VAL_RECORDS,
    COMBO_ZIPS,
    EXPORT,
    KAGGLE,
    REPORTS,
    aihub_val_zip,
    save_table,
    ts_zip,
)
from etl.classifier.validate_source_labels import (
    image_fingerprint,
    validate_source_labels,
)

TRAINING_PACKAGE = EXPORT / "training_package_classifier_v2"
CONDITIONS = {"back_color": "연회색 배경", "light_color": "주백색"}
ANGLES = {70, 75, 90}
PHASH_DISTANCE = 5  # 1차 검수의 유사 이미지 기준과 같다
SAMPLE_COLUMNS = [
    "sample_id",
    "split",
    "class_index",
    "code",
    "category_id",
    "item_seq",
    "name",
    "image_id",
    "image_path",
    "bbox_xywh",
    "group_id",
    "drug_dir",
    "source",
    "near_dup_in_train",
    "crop_path",
]
EXCLUDED_COLUMNS = ["image_id", "sample_id", "code", "stage", "reason", "detail"]
REVIEW_FILES = ("position_suspects.csv", "class_suspects.csv")
REVIEW_REQUIRED = {
    "position_suspects.csv": [
        "sample_id",
        "image_id",
        "code",
        "판정",
        "조치",
        "메모",
        "검수",
    ],
    "class_suspects.csv": [
        "sample_id",
        "image_id",
        "code",
        "suspect_code",
        "판정",
        "조치",
        "메모",
        "검수",
    ],
}


@dataclass(frozen=True, slots=True)
class EvalPackageResult:
    package_dir: Path
    zip_path: Path
    eval_summary: dict[str, Any]
    excluded: pd.DataFrame


# ── 라벨 읽기 ────────────────────────────────────────────────────────────────
def read_val_labels(label_zip: Path) -> pd.DataFrame:
    """VL 조합 zip의 라벨을 bbox(약) 하나당 한 줄로 읽는다. 각도 사진만 (라벨 없는 _index 사진은 없음)."""
    rows = []
    with zipfile.ZipFile(label_zip) as archive:
        for member in archive.namelist():
            if not member.endswith(".json") or "/._" in member:
                continue
            stem = os.path.basename(member)[:-5]
            parts = stem.split("_")
            if len(parts) != 8:
                continue
            data = json.loads(archive.read(member))
            image = data["images"][0]
            for annotation in data["annotations"]:
                rows.append(
                    {
                        "image_id": stem,
                        "code": member.split("/")[-2],
                        "bbox": annotation.get("bbox"),
                        "item_seq": str(image.get("item_seq", "")),
                        "drug_dir": image.get("drug_dir") or "",
                        "back_color": image.get("back_color"),
                        "light_color": image.get("light_color"),
                        "camera_la": image.get("camera_la"),
                        "file_angle": parts[5],
                    }
                )
    labels = pd.DataFrame(rows)
    if labels.empty:
        raise ValueError(f"라벨을 찾지 못했습니다: {label_zip}")
    labels["group_id"] = labels.image_id.str.split("_").str[0]
    return labels


def assign_sample_ids(labels: pd.DataFrame) -> pd.DataFrame:
    """sample_id = <사진 이름>_<파일명 속 약 코드 순번>. 다시 만들어도 같은 값이 나온다."""
    labels = labels.copy()

    def order(row: Any) -> int:
        codes = [f"K-{c}" for c in row.group_id.split("-")[1:]]
        return codes.index(row.code) if row.code in codes else -1

    labels["order"] = [order(row) for row in labels.itertuples()]
    if (labels.order < 0).any():
        bad = labels.loc[labels.order < 0, "image_id"].head(3).tolist()
        raise ValueError(f"파일명에 없는 약 코드의 라벨이 있습니다: {bad}")
    labels["sample_id"] = labels.image_id + "_" + labels.order.astype(str)
    if not labels.sample_id.is_unique:
        raise ValueError("sample_id가 중복됩니다 (한 사진에 같은 약 라벨이 두 개)")
    return labels


def check_conditions(labels: pd.DataFrame, class_map: dict[str, Any]) -> pd.DataFrame:
    """촬영 조건이 다르거나 118종 밖인 사진의 제외 목록."""
    codes = {f"K-{entry['category_id']:06d}" for entry in class_map["classes"]}
    rows = []
    for row in labels.itertuples():
        reasons = []
        for column, expected in CONDITIONS.items():
            if getattr(row, column) != expected:
                reasons.append(f"{column}={getattr(row, column)}")
        if row.camera_la not in ANGLES or str(row.camera_la) != row.file_angle:
            reasons.append(f"camera_la={row.camera_la}, 파일명 각도={row.file_angle}")
        if reasons:
            rows.append(
                (
                    row.image_id,
                    row.sample_id,
                    row.code,
                    "촬영 조건 다름",
                    "; ".join(reasons),
                )
            )
        if row.code not in codes:
            rows.append(
                (row.image_id, row.sample_id, row.code, "118종 밖 클래스", row.code)
            )
    return pd.DataFrame(
        [
            {
                "image_id": i,
                "sample_id": s,
                "code": c,
                "stage": "조건",
                "reason": r,
                "detail": d,
            }
            for i, s, c, r, d in rows
        ],
        columns=EXCLUDED_COLUMNS,
    )


# ── 학습 데이터와 중복 ──────────────────────────────────────────────────────
def _image_index(image_zip: Path) -> dict[str, str]:
    with zipfile.ZipFile(image_zip) as archive:
        return {
            os.path.basename(member)[:-4]: member
            for member in archive.namelist()
            if member.endswith(".png") and "/._" not in member
        }


def hash_eval_images(image_zip: Path, image_ids: list[str]) -> pd.DataFrame:
    index = _image_index(image_zip)
    rows = []
    with zipfile.ZipFile(image_zip) as archive:
        for k, image_id in enumerate(image_ids, 1):
            with Image.open(io.BytesIO(archive.read(index[image_id]))) as image:
                image.load()
                pixel_md5, phash = image_fingerprint(image)
            rows.append({"image_id": image_id, "md5": pixel_md5, "phash": int(phash)})
            if k % 500 == 0:
                print(f"  평가 사진 hash {k}/{len(image_ids)}")
    return pd.DataFrame(rows)


def hash_training_images(training_package: Path, cache_path: Path) -> pd.DataFrame:
    """학습 패키지(train+val)가 쓴 원본 사진의 hash. 한 번 계산하면 cache_path에 저장해 다시 쓴다."""
    samples = pd.read_csv(training_package / "samples.csv", dtype=str)
    images = samples.drop_duplicates("image_id")[["image_id", "image_path", "split"]]
    if cache_path.exists():
        cached = pd.read_csv(
            cache_path, dtype={"image_id": str, "md5": str, "phash": "uint64"}
        )
        if set(cached.image_id) == set(images.image_id):
            return cached
    zips = {ts_zip(n).name: ts_zip(n) for n in COMBO_ZIPS}
    archives: dict[str, zipfile.ZipFile] = {}
    rows = []
    try:
        for k, row in enumerate(images.itertuples(), 1):
            if row.image_path.startswith("kaggle/"):
                image = Image.open(KAGGLE / "train_images" / f"{row.image_id}.png")
            else:
                _, zip_name, member = row.image_path.split("/", 2)
                archive = archives.setdefault(zip_name, zipfile.ZipFile(zips[zip_name]))
                image = Image.open(io.BytesIO(archive.read(member)))
            with image:
                image.load()
                pixel_md5, phash = image_fingerprint(image)
            rows.append(
                {
                    "image_id": row.image_id,
                    "split": row.split,
                    "md5": pixel_md5,
                    "phash": int(phash),
                }
            )
            if k % 500 == 0:
                print(f"  학습 사진 hash {k}/{len(images)}")
    finally:
        for archive in archives.values():
            archive.close()
    hashes = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_table(hashes, cache_path)
    return hashes


def find_training_duplicates(
    eval_hashes: pd.DataFrame,
    train_hashes: pd.DataFrame,
    max_distance: int = PHASH_DISTANCE,
) -> pd.DataFrame:
    """평가 사진마다 가장 닮은 학습 사진. 픽셀 동일이거나 pHash 거리가 max_distance 이하인 것만 반환."""
    columns = ["image_id", "train_image_id", "train_split", "kind", "phash_distance"]
    if eval_hashes.empty or train_hashes.empty:
        return pd.DataFrame(columns=columns)
    eval_bits = eval_hashes.phash.astype("uint64").to_numpy()
    train_bits = train_hashes.phash.astype("uint64").to_numpy()
    distance = np.bitwise_count(eval_bits[:, None] ^ train_bits[None, :])
    md5_to_train = dict(zip(train_hashes.md5, train_hashes.image_id, strict=False))
    split_of = dict(zip(train_hashes.image_id, train_hashes.split, strict=True))
    rows = []
    for i, row in enumerate(eval_hashes.itertuples()):
        if row.md5 in md5_to_train:
            train_id = md5_to_train[row.md5]
            kind, dist = "픽셀 동일", 0
        else:
            j = int(distance[i].argmin())
            dist = int(distance[i, j])
            if dist > max_distance:
                continue
            train_id, kind = train_hashes.image_id.iloc[j], "유사 (pHash)"
        rows.append(
            {
                "image_id": row.image_id,
                "train_image_id": train_id,
                "train_split": split_of[train_id],
                "kind": kind,
                "phash_distance": dist,
            }
        )
    return pd.DataFrame(rows, columns=columns)


# ── 2차 검수 판정 ───────────────────────────────────────────────────────────
def read_review_records(records_dir: Path) -> dict[str, pd.DataFrame] | None:
    """사람 판정 CSV. 둘 다 없으면 None (아직 2차 검수 전)."""
    found = {
        name: records_dir / name
        for name in REVIEW_FILES
        if (records_dir / name).exists()
    }
    if not found:
        return None
    frames = {}
    for name in REVIEW_FILES:
        if name in found:
            frames[name] = pd.read_csv(found[name], dtype=str, keep_default_na=False)
        else:
            frames[name] = pd.DataFrame(columns=REVIEW_REQUIRED[name])
    return frames


# ── 패키지 ──────────────────────────────────────────────────────────────────
def build_samples(
    labels: pd.DataFrame,
    class_map: dict[str, Any],
    image_zip: Path,
) -> pd.DataFrame:
    classes = {f"K-{entry['category_id']:06d}": entry for entry in class_map["classes"]}
    index = _image_index(image_zip)
    mismatched = labels[
        [
            str(classes[c]["item_seq"]) != s
            for c, s in zip(labels.code, labels.item_seq, strict=True)
        ]
    ]
    if len(mismatched):
        raise ValueError(
            f"라벨의 item_seq가 class_map과 다릅니다: {mismatched[['image_id', 'code', 'item_seq']].head(3).to_dict('records')}"
        )
    samples = pd.DataFrame(
        {
            "sample_id": labels.sample_id,
            "split": EVAL_SPLIT,
            "class_index": [classes[c]["class_index"] for c in labels.code],
            "code": labels.code,
            "category_id": labels.code.str[2:].astype(int),
            "item_seq": labels.item_seq,
            "name": [classes[c]["name"] for c in labels.code],
            "image_id": labels.image_id,
            "image_path": [
                f"aihub_val/{image_zip.name}/{index[i]}" for i in labels.image_id
            ],
            "bbox_xywh": [json.dumps([float(v) for v in b]) for b in labels.bbox],
            "group_id": labels.group_id,
            "drug_dir": labels.drug_dir,
            "source": "aihub_val",
            "near_dup_in_train": False,  # 3단계에서 학습과 닮은 사진을 표시한다
        }
    )
    samples["crop_path"] = "crops/" + samples.code + "/" + samples.sample_id + ".png"
    return samples.sort_values("sample_id").reset_index(drop=True)


def write_package(
    samples: pd.DataFrame,
    class_map: dict[str, Any],
    excluded: pd.DataFrame,
    image_zip: Path,
    package_dir: Path,
) -> dict[str, Any]:
    index = _image_index(image_zip)
    package_dir.mkdir(parents=True)
    with zipfile.ZipFile(image_zip) as archive:
        for k, (image_id, group) in enumerate(
            samples.groupby("image_id", sort=False), 1
        ):
            with Image.open(io.BytesIO(archive.read(index[image_id]))) as source:
                image = source.convert("RGB")
            for row in group.itertuples():
                out = package_dir / row.crop_path
                out.parent.mkdir(parents=True, exist_ok=True)
                cut_pill(image, json.loads(row.bbox_xywh), CROP_MARGIN).save(out)
            if k % 500 == 0:
                print(f"  crop {k}/{samples.image_id.nunique()}장")
    samples[SAMPLE_COLUMNS].to_csv(
        package_dir / "samples.csv", index=False, encoding="utf-8"
    )
    (package_dir / "class_map.json").write_text(
        json.dumps(class_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_table(excluded, package_dir / "excluded.csv")
    summary = make_eval_summary(read_samples(package_dir), class_map["num_classes"])
    (package_dir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def write_report(
    *,
    path: Path,
    labels: pd.DataFrame,
    samples: pd.DataFrame,
    excluded: pd.DataFrame,
    duplicates: pd.DataFrame,
    summary: dict[str, Any],
    training_package: Path,
    reviewed: bool,
    label_zip: Path,
    image_zip: Path,
) -> None:
    train = pd.read_csv(training_package / "samples.csv", dtype=str)
    train_counts = train[train.split == "train"].code.value_counts()
    image_reasons = excluded.groupby("reason").image_id.nunique()
    lines = [
        "# AI Hub val 외부 평가셋 (분류기)",
        "",
        "`python -m etl.classifier.aihub_val.build_eval_package`가 만든다. 학습에는 쓰지 않는 평가 전용 데이터다.",
        "",
        f"- 실행일: {date.today()}",
        f"- 원본: `{label_zip.name}` (라벨), `{image_zip.name}` (이미지)",
        f"- class_map·중복 검사 기준 학습 패키지: `{training_package.name}`",
        f"- 2차 라벨 검수(사람 판정): {'반영함' if reviewed else '**아직 반영 전** (records/ 판정 없음)'}",
        "",
        "## 요약",
        "",
        "| 항목 | 원본 | 평가셋 |",
        "| --- | ---: | ---: |",
        f"| 사진 | {labels.image_id.nunique():,} | {summary['num_eval_images']:,} |",
        f"| crop(bbox) | {len(labels):,} | {summary['num_eval_samples']:,} |",
        f"| 조합 그룹 | {labels.group_id.nunique():,} | {summary['eval_group_count']:,} |",
        f"| 클래스 | {labels.code.nunique()} | {len(summary['eval_class_indices'])} |",
        "",
        f"- eval_hash: `{summary['eval_hash']}`",
        "- 크롭: 학습과 같은 `cut_pill` (bbox 사방 5% 여백 + 이미지 경계 clip), class_map은 학습 패키지 것을 그대로 복사",
        "",
        "## 제외",
        "",
        "| 단계 | 사유 | 사진 | crop |",
        "| --- | --- | ---: | ---: |",
    ]
    if excluded.empty:
        lines.append("| - | 없음 | 0 | 0 |")
    crops_per_image = labels.groupby("image_id").size()
    for (stage, reason), group in excluded.groupby(["stage", "reason"], sort=False):
        image_level = group[group.sample_id == ""]
        n_crops = int(
            crops_per_image.reindex(image_level.image_id.unique()).sum()
        ) + int((group.sample_id != "").sum())
        lines.append(f"| {stage} | {reason} | {group.image_id.nunique()} | {n_crops} |")
    lines += [
        "",
        "- 사진 하나라도 문제가 있으면 그 사진의 crop을 모두 뺀다 (AI Hub 조합 데이터의 1차 검수 규칙과 같음)",
        "- 한 사진이 여러 사유에 걸리면 사유마다 센다",
        "- 제외 목록: 패키지의 `excluded.csv` (사진·crop 단위, 사유)",
        "",
        "## 학습 데이터와 중복",
        "",
        "학습 패키지가 쓴 원본 사진(train+val)과 비교했다.",
        "",
        f"- 학습 패키지와 같은 파일명: {int(labels.image_id.isin(set(train.image_id)).groupby(labels.image_id).first().sum())}장",
        f"- 학습 패키지에 있는 조합 그룹의 사진: {int(labels.group_id.isin(set(train.group_id)).groupby(labels.image_id).first().sum())}장",
        f"- 픽셀 동일: {(duplicates.kind == '픽셀 동일').sum()}장 → 제외",
        f"- 유사 (pHash ≤ {PHASH_DISTANCE}): {(duplicates.kind != '픽셀 동일').sum()}장 → 제외하지 않고 `near_dup_in_train=True`로 표시"
        f" (평가셋에 남은 crop {int(samples.near_dup_in_train.astype(str).eq('True').sum()):,}개)",
        "",
        "유사 사진은 조합(약 구성)이 다른 **다른 사진**이다. AI Hub는 알약을 정해진 자리에 놓고 찍어서, 같은 알약이 같은 자리·방향에 놓이면 사진 전체가 닮는다.",
        "학습 패키지 val도 같은 기준(pHash ≤ 5)으로 빼지 않고 표시만 했으므로 같은 규칙을 따른다. 평가할 때 전체 점수와 표시를 뺀 점수를 함께 본다.",
        "",
    ]
    if len(duplicates):
        lines += [
            f"pHash 거리별 사진 수: {duplicates.phash_distance.value_counts().sort_index().to_dict()}. "
            "쌍 목록: `data/classifier/aihub_val/duplicates_with_training.csv`",
            "",
        ]
    counts = samples.groupby(["code", "name"]).size().rename("eval").reset_index()
    counts["train"] = counts.code.map(train_counts).fillna(0).astype(int)
    counts = counts.sort_values(["eval", "code"], ascending=[False, True])
    lines += [
        "## 클래스별 crop 수",
        "",
        f"평가셋에 있는 {len(counts)}종. 나머지 {summary['num_classes'] - len(counts)}종은 이 평가셋으로 측정할 수 없다.",
        "개수가 고르지 않으므로 Macro F1(평가셋에 있는 클래스 평균)을 주 지표로 본다.",
        "",
        "| 코드 | 품목 | 평가셋 crop | 학습 패키지 train crop |",
        "| --- | --- | ---: | ---: |",
    ]
    lines += [
        f"| {r.code} | {r.name} | {r.eval} | {r.train} |" for r in counts.itertuples()
    ]
    lines += [
        "",
        f"- 20개 미만 클래스: {(counts['eval'] < 20).sum()}종",
        f"- 사진 면 (`drug_dir`): {samples.drug_dir.value_counts().to_dict()}",
        f"- 제외 사유별 사진 수: {image_reasons.to_dict() if len(image_reasons) else '없음'}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_aihub_val_eval_package(
    *,
    label_zip: Path | None = None,
    image_zip: Path | None = None,
    training_package: Path = TRAINING_PACKAGE,
    package_dir: Path = AIHUB_VAL_PACKAGE,
    meta_dir: Path = AIHUB_VAL_META,
    records_dir: Path = AIHUB_VAL_RECORDS,
    report_dir: Path = REPORTS,
    skip_png: bool = False,
    overwrite: bool = False,
) -> EvalPackageResult:
    label_zip = label_zip or aihub_val_zip("VL_1")
    image_zip = image_zip or aihub_val_zip("VS_1")
    if package_dir.exists() and not overwrite:
        raise FileExistsError(
            f"평가 패키지가 이미 있습니다 (--overwrite로 교체): {package_dir}"
        )
    class_map = json.loads(
        (training_package / "class_map.json").read_text(encoding="utf-8")
    )

    print("1. 1차 라벨 검수")
    issues = validate_source_labels(
        skip_png=skip_png,
        label_zips=[label_zip],
        image_zips=[image_zip],
        include_kaggle=False,
        meta_dir=meta_dir,
        report_path=report_dir / "aihub_val_validation_report.md",
        target_description=f"AI Hub val 조합 `{label_zip.name}`",
    )
    labels = assign_sample_ids(read_val_labels(label_zip))
    image_issues = issues[(issues.action == "이미지 제외") & (issues.image != "")]
    excluded = [
        pd.DataFrame(
            {
                "image_id": image_issues.image.str.removesuffix(".png"),
                "sample_id": "",
                "code": image_issues.code,
                "stage": "1차 라벨 검수",
                "reason": image_issues.check,
                "detail": image_issues.detail,
            }
        )
    ]

    print("2. 촬영 조건·클래스")
    excluded.append(check_conditions(labels, class_map))

    print("3. 학습 데이터와 중복")
    eval_hashes = hash_eval_images(image_zip, sorted(labels.image_id.unique()))
    train_hashes = hash_training_images(
        training_package, meta_dir / "training_image_hashes.csv"
    )
    duplicates = find_training_duplicates(eval_hashes, train_hashes)
    save_table(duplicates, meta_dir / "duplicates_with_training.csv")
    exact = duplicates[duplicates.kind == "픽셀 동일"]
    near_images = set(duplicates.loc[duplicates.kind != "픽셀 동일", "image_id"])
    excluded.append(
        pd.DataFrame(
            {
                "image_id": exact.image_id,
                "sample_id": "",
                "code": "",
                "stage": "학습 데이터와 중복",
                "reason": exact.kind,
                "detail": [
                    f"{r.train_image_id} ({r.train_split}), pHash 거리 {r.phash_distance}"
                    for r in exact.itertuples()
                ],
            }
        )
    )
    excluded = pd.concat(excluded, ignore_index=True)[EXCLUDED_COLUMNS]
    dropped_images = set(excluded.image_id)
    samples = build_samples(
        labels[~labels.image_id.isin(dropped_images)], class_map, image_zip
    )
    samples["near_dup_in_train"] = samples.image_id.isin(near_images)

    print("4. 2차 라벨 검수 판정")
    reviews = read_review_records(records_dir)
    if reviews is not None:
        actions = build_label_actions(
            samples, reviews["position_suspects.csv"], reviews["class_suspects.csv"]
        )
        reviewed_samples = apply_label_actions(
            samples.astype({"class_index": str, "category_id": str}), actions, class_map
        )
        dropped = actions[actions.action.isin({"이미지 제외", "bbox 제외"})]
        excluded = pd.concat(
            [
                excluded,
                pd.DataFrame(
                    {
                        "image_id": dropped.image_id,
                        "sample_id": dropped.sample_id,
                        "code": dropped.code,
                        "stage": "2차 라벨 검수",
                        "reason": dropped.action,
                        "detail": dropped["메모"],
                    }
                ),
            ],
            ignore_index=True,
        )
        samples = reviewed_samples
        save_table(actions, meta_dir / "review_actions.csv")
    excluded = excluded.sort_values(["stage", "image_id", "sample_id"]).reset_index(
        drop=True
    )

    print("5. 패키지 쓰기")
    with tempfile.TemporaryDirectory(
        dir=package_dir.parent if package_dir.parent.exists() else None
    ) as tmp:
        staged = Path(tmp) / package_dir.name
        summary = write_package(samples, class_map, excluded, image_zip, staged)
        loader, _, _ = build_classifier_eval_loader(
            staged, batch_size=32, device=torch.device("cpu")
        )
        batch = next(iter(loader))
        if tuple(batch["images"].shape[1:]) != (3, 224, 224):
            raise ValueError(
                f"평가 로더 출력 크기가 다릅니다: {tuple(batch['images'].shape)}"
            )
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), package_dir)
    zip_path = Path(shutil.make_archive(str(package_dir), "zip", package_dir))

    write_report(
        path=report_dir / "aihub_val_eval_package_report.md",
        labels=labels,
        samples=read_samples(package_dir),
        excluded=excluded,
        duplicates=duplicates,
        summary=summary,
        training_package=training_package,
        reviewed=reviews is not None,
        label_zip=label_zip,
        image_zip=image_zip,
    )
    print(
        f"평가셋: 사진 {summary['num_eval_images']}장, crop {summary['num_eval_samples']}개, "
        f"클래스 {len(summary['eval_class_indices'])}종 / 제외 사진 {excluded.image_id.nunique()}장"
    )
    print("저장:", package_dir, zip_path)
    return EvalPackageResult(package_dir, zip_path, summary, excluded)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-zip",
        type=Path,
        help="VL_1_조합.zip (기본: AIHUB_VAL_DIR 아래에서 찾음)",
    )
    parser.add_argument(
        "--image-zip",
        type=Path,
        help="VS_1_조합.zip (기본: AIHUB_VAL_DIR 아래에서 찾음)",
    )
    parser.add_argument("--training-package", type=Path, default=TRAINING_PACKAGE)
    parser.add_argument("--package-dir", type=Path, default=AIHUB_VAL_PACKAGE)
    parser.add_argument("--records-dir", type=Path, default=AIHUB_VAL_RECORDS)
    parser.add_argument(
        "--skip-png", action="store_true", help="1차 검수의 PNG 검사 생략"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build_aihub_val_eval_package(
        label_zip=args.label_zip,
        image_zip=args.image_zip,
        training_package=args.training_package,
        package_dir=args.package_dir,
        records_dir=args.records_dir,
        skip_png=args.skip_png,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
