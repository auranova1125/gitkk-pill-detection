"""이미지와 박스를 모델 입력으로 바꾸고 배치를 구성한다."""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .settings import load_settings
from .data_records import read_records

if TYPE_CHECKING:
    from detector import DetectorConfig

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
                f"실제 이미지와 JSON 크기 불일치: {record['original_file_name']}"
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


def build_detector_inputs(*, config: DetectorConfig, device):
    """팀원의 학습 코드가 호출하는 공통 함수. 데이터는 다시 나누지 않는다."""
    return _build_detector_inputs(config=config, device=device, settings=load_settings())


def _build_detector_inputs(*, config, device, settings):
    records_by_split, label_map, summary = read_records(settings)
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

    summary = dict(summary, train_augmentation=dict(augmentation), validation_augmentation="none")

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
