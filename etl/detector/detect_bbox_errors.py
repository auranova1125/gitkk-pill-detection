"""미검수 탐지 이미지 입력표와 bbox 의심 후보·검수 그림 생성."""

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

from etl.classifier.label_audit.detect_bbox_errors import detect_bbox_errors
from etl.detector.build_detector_v2 import local_settings, read_records
from utils.io import write_json


def prepare_samples(package_dir, classifier_samples, output_dir):
    """이미 검수한 원본 사진을 제외하고 bbox 하나당 입력 행 하나를 만든다."""
    with classifier_samples.open(encoding="utf-8-sig", newline="") as stream:
        checked = {row["image_id"] for row in csv.DictReader(stream)}
    records, _, summary = read_records(local_settings(package_dir))
    rows = []
    paths = {}
    for split, images in records.items():
        for image in images:
            image_id = re.sub(
                r"^(codeit_|source_\d+_)", "", Path(image["image_path"]).stem
            )
            if image_id in checked:
                continue
            if image["dataset"] != "select":
                raise ValueError(
                    f"미검수 목록에 AI Hub가 아닌 이미지가 있습니다: {image_id}"
                )
            name = f"{image_id}.png"
            if name in paths:
                raise ValueError(f"원본 이미지 식별자 중복: {image_id}")
            paths[name] = image["image_path"]
            for annotation_id, box, code in zip(
                image["annotation_ids"],
                image["bbox_xywh"],
                image["drug_codes"],
                strict=True,
            ):
                rows.append(
                    {
                        "sample_id": f"{image_id}__{annotation_id}",
                        "image_id": image_id,
                        "code": code,
                        "name": "",
                        "split": split,
                        "source": "aihub",
                        "bbox_xywh": json.dumps(box),
                    }
                )
    if not rows:
        raise ValueError("검사할 bbox가 없습니다.")
    samples = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_dir / "samples.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "source_images.json", paths)
    write_json(
        output_dir / "input_summary.json",
        {
            "package_dir": str(package_dir.resolve()),
            "split_hash": summary["split_hash"],
            "classifier_samples": str(classifier_samples.resolve()),
            "already_checked_images": len(checked),
            "images": len(paths),
            "bbox": len(rows),
            "images_by_split": samples.groupby("split")["image_id"].nunique().to_dict(),
            "bbox_by_split": samples["split"].value_counts().to_dict(),
        },
    )
    print(f"검사 입력: 사진 {len(paths)}장, bbox {len(rows)}개", flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path("datasets/select/v2"))
    parser.add_argument("--classifier-samples", type=Path, required=True)
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("datasets/select/position_audit")
    )
    parser.add_argument(
        "--records-dir", type=Path, default=Path("etl/detector/label_audit")
    )
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    input_dir = args.audit_dir / "input"
    paths = prepare_samples(args.package_dir, args.classifier_samples, input_dir)
    if not args.prepare_only:
        suspects = detect_bbox_errors(
            package_dir=input_dir,
            records_dir=args.records_dir,
            output_dir=args.audit_dir / "output",
            processes=args.processes,
            source_image_paths=paths,
        )
        print(f"위치 오류 후보 bbox {len(suspects)}개", flush=True)
        print(
            suspects.groupby("priority")
            .agg(bbox=("sample_id", "size"), images=("image_id", "nunique"))
            .to_string(),
            flush=True,
        )


if __name__ == "__main__":
    main()
