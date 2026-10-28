
"""고정 분할을 사용하는 단일 클래스 알약 탐지 데이터 구성."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, get_worker_info

if TYPE_CHECKING:
    from detector import DetectorConfig


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def original_file_name(file_name):
    name = Path(file_name).name
    match = re.search(r"(K-\d{6}.*)$", name)

    if match is None:
        raise ValueError(f"원본 파일명을 확인할 수 없습니다: {name}")

    return match.group(1)


def canonical_group(file_name):
    prefix = original_file_name(file_name).split("_", 1)[0]
    codes = prefix.removeprefix("K-").split("-")

    if not codes or not all(
        re.fullmatch(r"\d{6}", code) for code in codes
    ):
        raise ValueError(f"알약 조합 형식 오류: {file_name}")

    return "K-" + "-".join(sorted(codes))


def collate_detector_samples(samples):
    return {
        "images": [sample["image"] for sample in samples],
        "targets": [sample["target"] for sample in samples],
        "meta": [sample["meta"] for sample in samples],
    }


def seed_detector_worker(worker_id):
    info = get_worker_info()
    if info is not None:
        info.dataset.rng.seed(torch.initial_seed())


class PillDetectionDataset(Dataset):
    def __init__(self, records, augmentation, seed):
        self.records = records
        self.augmentation = dict(augmentation)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")

        width, height = image.size

        if (width, height) != (record["width"], record["height"]):
            raise ValueError(
                f"실제 이미지와 JSON 크기 불일치: {record['file_name']}"
            )

        boxes = torch.tensor(
            record["boxes"], dtype=torch.float32
        ).reshape(-1, 4)

        horizontal = self.augmentation["horizontal_flip"]
        vertical = self.augmentation["vertical_flip"]

        if horizontal > 0 and self.rng.random() < horizontal:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            left = boxes[:, 0].clone()
            right = boxes[:, 2].clone()
            boxes[:, 0] = width - right
            boxes[:, 2] = width - left

        if vertical > 0 and self.rng.random() < vertical:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            top = boxes[:, 1].clone()
            bottom = boxes[:, 3].clone()
            boxes[:, 1] = height - bottom
            boxes[:, 3] = height - top

        brightness = self.augmentation["brightness"]
        contrast = self.augmentation["contrast"]

        if brightness > 0:
            image = ImageEnhance.Brightness(image).enhance(
                self.rng.uniform(1 - brightness, 1 + brightness)
            )

        if contrast > 0:
            image = ImageEnhance.Contrast(image).enhance(
                self.rng.uniform(1 - contrast, 1 + contrast)
            )

        image_tensor = torch.from_numpy(
            np.array(image, dtype=np.uint8, copy=True)
        ).permute(2, 0, 1).contiguous().to(torch.float32).div_(255)

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)

        if not torch.all(boxes[:, 2:] > boxes[:, :2]):
            raise ValueError(
                f"변환 후 박스 면적 오류: 이미지 {record['image_id']}"
            )

        count = len(boxes)

        target = {
            "boxes": boxes,
            "labels": torch.ones(count, dtype=torch.int64),
            "image_id": record["image_id"],
            "area": (
                (boxes[:, 2] - boxes[:, 0])
                * (boxes[:, 3] - boxes[:, 1])
            ),
            "iscrowd": torch.zeros(count, dtype=torch.int64),
        }

        meta = {
            "image_id": record["image_id"],
            "source_image_id": record["source_image_id"],
            "file_name": record["original_file_name"],
            "image_path": record["image_path"],
            "group_id": record["group_id"],
            "original_size": [height, width],
            "source_annotation_count": count,
            "annotation_ids": list(record["annotation_ids"]),
            "bbox_xywh": [list(box) for box in record["bbox_xywh"]],
            "source_category_ids": list(record["category_ids"]),
            "drug_codes": list(record["drug_codes"]),
            "dataset": record["dataset"],
            "transformed_boxes_xyxy": boxes.tolist(),
        }

        return {
            "image": image_tensor,
            "target": target,
            "meta": meta,
        }


def validate_batch(batch):
    for key in ["images", "targets", "meta"]:
        if not isinstance(batch.get(key), list):
            raise ValueError(f"배치의 {key}는 리스트여야 합니다.")

    if not batch["images"]:
        raise ValueError("빈 배치입니다.")

    if not (
        len(batch["images"])
        == len(batch["targets"])
        == len(batch["meta"])
    ):
        raise ValueError("배치 리스트 길이가 일치하지 않습니다.")

    for image, target, meta in zip(
        batch["images"], batch["targets"], batch["meta"]
    ):
        assert image.device.type == "cpu", "이미지는 CPU에 있어야 합니다."
        assert image.dtype == torch.float32, "이미지 자료형 오류"
        assert image.ndim == 3 and image.shape[0] == 3, "이미지 형태 오류"
        assert image.shape[1] > 0 and image.shape[2] > 0
        assert torch.isfinite(image).all(), "이미지 숫자 오류"
        assert image.min() >= 0 and image.max() <= 1, "이미지 범위 오류"

        height, width = image.shape[-2:]
        boxes = target["boxes"]
        count = len(boxes)

        assert count > 0, "박스 없는 이미지"
        assert boxes.dtype == torch.float32
        assert boxes.shape == (count, 4)
        assert torch.isfinite(boxes).all()
        assert torch.all(boxes[:, 0] >= 0)
        assert torch.all(boxes[:, 1] >= 0)
        assert torch.all(boxes[:, 2] <= width)
        assert torch.all(boxes[:, 3] <= height)
        assert torch.all(boxes[:, 2:] > boxes[:, :2])

        assert target["labels"].dtype == torch.int64
        assert torch.all(target["labels"] == 1)
        assert target["area"].dtype == torch.float32
        assert target["iscrowd"].dtype == torch.int64
        assert torch.all(target["iscrowd"] == 0)

        for field in ["labels", "area", "iscrowd"]:
            assert target[field].shape == (count,)

        expected_area = (
            (boxes[:, 2] - boxes[:, 0])
            * (boxes[:, 3] - boxes[:, 1])
        )
        assert torch.allclose(target["area"], expected_area)

        assert type(target["image_id"]) is int
        assert target["image_id"] >= 0
        assert target["image_id"] == meta["image_id"]

        assert count == meta["source_annotation_count"]
        assert count == len(meta["annotation_ids"])
        assert count == len(meta["bbox_xywh"])
        assert len(set(meta["annotation_ids"])) == count

        assert isinstance(meta["source_image_id"], str)
        assert meta["file_name"] and meta["image_path"] and meta["group_id"]

    return True


def build_detector_inputs(*, config: DetectorConfig, device):
    settings = read_json(
        Path(__file__).with_name("detector_data_settings.json")
    )

    package = Path(settings["package_dir"]).expanduser().resolve()
    roots = {
        name: Path(value).expanduser().resolve()
        for name, value in settings["image_roots"].items()
    }

    with (package / "split_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        manifest_rows = list(csv.DictReader(file))

    manifest = {}

    for row in manifest_rows:
        image_id = int(row["image_id"])

        if image_id in manifest:
            raise ValueError(f"분할 목록의 이미지 ID 중복: {image_id}")

        manifest[image_id] = row

    all_image_ids = set()
    all_annotation_ids = set()
    records_by_split = {}
    clipped_annotation_ids = []

    for split in ["train", "validation"]:
        labels = read_json(package / "annotations" / f"{split}.json")
        split_ids = set()
        annotation_groups = defaultdict(list)

        for image in labels["images"]:
            image_id = image["id"]

            if (
                type(image_id) is not int
                or image_id < 0
                or image_id in all_image_ids
            ):
                raise ValueError(f"이미지 ID 오류 또는 중복: {image_id}")

            all_image_ids.add(image_id)
            split_ids.add(image_id)

        for annotation in labels["annotations"]:
            annotation_id = annotation["id"]

            if (
                type(annotation_id) is not int
                or annotation_id in all_annotation_ids
            ):
                raise ValueError(
                    f"annotation ID 오류 또는 중복: {annotation_id}"
                )

            if annotation["image_id"] not in split_ids:
                raise ValueError(
                    f"이미지에 연결되지 않은 annotation: {annotation_id}"
                )

            all_annotation_ids.add(annotation_id)
            annotation_groups[annotation["image_id"]].append(annotation)

        records = []

        for image in labels["images"]:
            image_id = image["id"]

            if image_id not in manifest:
                raise ValueError(f"분할 목록에 없는 이미지: {image_id}")

            row = manifest[image_id]

            if row["split"] != split:
                raise ValueError(f"분할 배정 불일치: {image_id}")

            width = int(image["width"])
            height = int(image["height"])

            if width <= 0 or height <= 0:
                raise ValueError(f"이미지 크기 오류: {image_id}")

            portable = str(image["file_name"]).replace("\\", "/")
            dataset, relative = portable.split("/", 1)
            relative_path = Path(relative)

            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"허용되지 않는 이미지 경로: {portable}")

            if dataset not in roots:
                raise ValueError(f"이미지 기준 폴더 설정 없음: {dataset}")

            if (
                dataset != row["dataset"]
                or portable != row["image_path"]
            ):
                raise ValueError(f"이미지 경로 기록 불일치: {image_id}")

            group = canonical_group(portable)

            if group != row["group_id"]:
                raise ValueError(f"조합 그룹 기록 불일치: {image_id}")

            actual_path = roots[dataset] / relative_path

            if not actual_path.is_file():
                raise FileNotFoundError(actual_path)

            annotations = annotation_groups[image_id]

            if not annotations:
                raise ValueError(f"박스 없는 이미지: {image_id}")

            if len(annotations) != int(row["annotation_count"]):
                raise ValueError(f"annotation 개수 불일치: {image_id}")

            boxes = []
            source_boxes = []

            for annotation in annotations:
                annotation_id = annotation["id"]
                raw_box = annotation["bbox"]

                if len(raw_box) != 4:
                    raise ValueError(f"bbox 원소 수 오류: {annotation_id}")

                x, y, w, h = map(float, raw_box)

                if not all(math.isfinite(v) for v in [x, y, w, h]):
                    raise ValueError(f"bbox 숫자 오류: {annotation_id}")

                if w <= 0 or h <= 0:
                    raise ValueError(f"bbox 크기 오류: {annotation_id}")

                if annotation.get("iscrowd", 0) != 0:
                    raise ValueError(
                        f"iscrowd가 0이 아닙니다: {annotation_id}"
                    )

                if annotation.get("ignore", 0) != 0:
                    raise ValueError(
                        f"ignore가 0이 아닙니다: {annotation_id}"
                    )

                before = [x, y, x + w, y + h]
                box = [
                    min(max(x, 0), width),
                    min(max(y, 0), height),
                    min(max(x + w, 0), width),
                    min(max(y + h, 0), height),
                ]

                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(
                        f"경계 처리 후 박스 면적이 0입니다: {annotation_id}"
                    )

                if box != before:
                    clipped_annotation_ids.append(annotation_id)

                boxes.append(box)
                source_boxes.append([x, y, w, h])

            records.append({
                "image_id": image_id,
                "source_image_id": str(row["original_image_id"]),
                "original_file_name": original_file_name(portable),
                "portable_path": portable,
                "image_path": str(actual_path.resolve()),
                "dataset": dataset,
                "group_id": group,
                "width": width,
                "height": height,
                "boxes": boxes,
                "bbox_xywh": source_boxes,
                "annotation_ids": [a["id"] for a in annotations],
                "category_ids": [int(a["category_id"]) for a in annotations],
                "drug_codes": [
                    a.get("drug_code", f"K-{int(a['category_id']):06d}")
                    for a in annotations
                ],
            })

        records_by_split[split] = sorted(
            records, key=lambda record: record["image_id"]
        )

    if set(manifest) != all_image_ids:
        raise ValueError("분할 목록과 라벨의 전체 이미지 구성이 다릅니다.")

    groups = {
        split: {record["group_id"] for record in records}
        for split, records in records_by_split.items()
    }

    overlap = groups["train"] & groups["validation"]

    if overlap:
        raise ValueError(
            f"Train과 Validation의 조합 그룹 중복: {len(overlap)}개"
        )

    label_map = {
        "num_foreground_classes": 1,
        "num_model_classes": 2,
        "background_label": 0,
        "classes": [{"label": 1, "name": "pill"}],
    }

    hash_records = [
        {
            "split": split,
            "image_id": record["image_id"],
            "group_id": record["group_id"],
            "file_name": record["portable_path"],
            "annotation_ids": record["annotation_ids"],
            "bbox_xywh": record["bbox_xywh"],
            "category_ids": record["category_ids"],
        }
        for split in ["train", "validation"]
        for record in records_by_split[split]
    ]

    split_hash = hashlib.sha256(
        json.dumps(
            {"version": "fixed-pill-data-v1", "records": hash_records},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    augmentation = settings["augmentation"]
    required_options = {
        "horizontal_flip", "vertical_flip", "brightness", "contrast"
    }

    if set(augmentation) != required_options:
        raise ValueError("증강 설정 항목이 올바르지 않습니다.")

    for name, value in augmentation.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"증강 설정의 숫자 오류: {name}")

        upper = 1.0 if name.endswith("flip") else 0.3

        if not 0 <= value <= upper:
            raise ValueError(f"증강 설정 범위 오류: {name}")

    summary = {
        "group_overlap_count": 0,
        "split_hash": split_hash,
        "clipped_box_count": len(clipped_annotation_ids),
        "clipped_annotation_ids": clipped_annotation_ids,
        "train_augmentation": dict(augmentation),
        "validation_augmentation": "none",
    }

    for split, name in [("train", "train"), ("validation", "val")]:
        records = records_by_split[split]
        counts = [len(record["boxes"]) for record in records]

        if not counts:
            raise ValueError(f"{split}: 이미지가 없습니다.")

        instance_count = sum(counts)

        summary.update({
            f"num_{name}_images": len(records),
            f"num_{name}_source_annotations": instance_count,
            f"num_{name}_instances": instance_count,
            f"{name}_group_count": len(groups[split]),
            f"{name}_empty_image_count": 0,
            f"{name}_max_instances_per_image": max(counts),
            f"{name}_label_counts": {"1": instance_count},
        })

    workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"
    loaders = []

    for split, offset in [("train", 0), ("validation", 1)]:
        transforms = (
            augmentation if split == "train"
            else {name: 0.0 for name in augmentation}
        )

        dataset = PillDetectionDataset(
            records_by_split[split],
            transforms,
            config.seed + offset,
        )

        generator = torch.Generator()
        generator.manual_seed(config.seed + offset)

        loaders.append(DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_detector_samples,
            worker_init_fn=seed_detector_worker,
            generator=generator,
        ))

    return loaders[0], loaders[1], label_map, summary
