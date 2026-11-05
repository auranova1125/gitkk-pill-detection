"""검증된 고정 탐지 패키지를 YOLO 입력으로 변환한다. 원본은 수정하지 않는다."""

import json
from pathlib import Path

import yaml
from PIL import Image

from data_loading.detector_package import load_detector_package
from utils.io import write_json

V2_SPLIT_HASH = "ac6618cdf2fd69352b0d3848b69064eadbc6527d6153795c3cc2de1c2dfa7d32"


def encode_boxes(record: dict) -> str:
    """검증된 xyxy를 단일 클래스 0의 정규화 cxcywh로 바꾼다."""
    width, height = record["width"], record["height"]
    lines = []
    for x1, y1, x2, y2 in record["boxes"]:
        lines.append(
            f"0 {(x1 + x2) / (2 * width):.12g} "
            f"{(y1 + y2) / (2 * height):.12g} "
            f"{(x2 - x1) / width:.12g} {(y2 - y1) / height:.12g}"
        )
    return "\n".join(lines) + "\n"


def export_yolo_package(
    settings: dict, destination: Path, *, expected_hash: str = V2_SPLIT_HASH
) -> Path:
    """전체 패키지 검증 후 ID 기반 symlink와 라벨·추적 metadata를 만든다.

    중단/완료 여부와 관계없이 기존 destination은 재사용하지 않는다.
    Colab에서는 원본 패키지를 /content로 복사한 후 호출한다.
    """
    records, label_map, summary = load_detector_package(settings)
    if summary["split_hash"] != expected_hash:
        raise ValueError("요청한 데이터 split_hash와 실제 패키지가 다릅니다.")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    trace = []
    for split, samples in records.items():
        yolo_split = "train" if split == "train" else "val"
        images = destination / "images" / yolo_split
        labels = destination / "labels" / yolo_split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for record in samples:
            source = Path(record["image_path"]).resolve()
            with Image.open(source) as image:
                if image.size != (record["width"], record["height"]):
                    raise ValueError(f"실제 이미지 크기 불일치: {source}")
            name = f"{record['image_id']}{source.suffix.lower()}"
            (images / name).symlink_to(source)
            (labels / f"{record['image_id']}.txt").write_text(
                encode_boxes(record), encoding="utf-8"
            )
            trace.append(dict(record, split=split, yolo_image=name))
    write_json(destination / "source_records.json", trace)
    write_json(destination / "detector_label_map.json", label_map)
    write_json(destination / "detector_data_summary.json", summary)
    write_json(destination / "yolo_label_map.json", {"0": "pill"})
    (destination / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(destination),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "pill"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_json(
        destination / "export_complete.json",
        {
            "split_hash": summary["split_hash"],
            "source_settings": settings,
            "class_mapping": {"repository_pill": 1, "yolo_pill": 0},
        },
    )
    return destination / "data.yaml"


def verify_export(data_yaml: Path) -> dict:
    """학습 직전 원본과 변환된 전체 라벨·링크를 대조한다."""
    root = Path(data_yaml).resolve().parent
    complete = json.loads((root / "export_complete.json").read_text())
    records, _, summary = load_detector_package(complete["source_settings"])
    if summary["split_hash"] != complete["split_hash"]:
        raise ValueError("변환 이후 원본 split_hash가 변경됐습니다.")
    for split, samples in records.items():
        name = "train" if split == "train" else "val"
        image_dir, label_dir = root / "images" / name, root / "labels" / name
        if len(list(image_dir.iterdir())) != len(samples):
            raise ValueError("변환 이미지 수 불일치")
        if len(list(label_dir.glob("*.txt"))) != len(samples):
            raise ValueError("변환 라벨 수 불일치")
        for record in samples:
            source = Path(record["image_path"]).resolve()
            image = image_dir / f"{record['image_id']}{source.suffix.lower()}"
            if not image.is_file() or image.resolve() != source:
                raise ValueError(f"변환 이미지 연결 오류: {image}")
            if (label_dir / f"{record['image_id']}.txt").read_text() != encode_boxes(
                record
            ):
                raise ValueError(f"변환 bbox 불일치: {record['image_id']}")
    spec = yaml.safe_load(Path(data_yaml).read_text())
    if spec != {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "pill"},
    }:
        raise ValueError("data.yaml이 검증된 변환 패키지와 다릅니다.")
    return summary
