"""경구약제 분류기 Dataset과 DataLoader factory.

학습 노트북:
    data_factory_path = "data_loading.classifier:build_classifier_inputs"
    config.class_map_path    = "<training_package_classifier.zip을 푼 폴더>/class_map.json"
    config.data_summary_path = "<training_package_classifier.zip을 푼 폴더>/data_summary.json"

class_map.json과 같은 폴더에 samples.csv와 crops/가 있어야 한다.
crops/의 PNG는 원본 사진에서 bbox 사방 5% 여백을 더해 이미지 경계로 clip해 자른 직사각형이다 (명세서 6.1의 1~3).
이후 처리(학습: 회전 → 중앙 정사각형 padding, 공통: 224 resize → normalize)는 이 파일에서 한다.

탐지 결과를 분류할 때 (명세서 8절):
    from data_loading.classifier import detection_transform
    from data_loading.image_transforms import crop_pill
    image = detection_transform()(crop_pill(original_image, bbox_xywh))   # float32 [3, 224, 224]

입력 검증:  python -m data_loading.classifier <class_map.json 경로>
"""

import random
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data_loading.classifier_package import load_classifier_package
from data_loading.image_transforms import (
    CLASSIFIER_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_classifier_inference_transform,
    build_classifier_train_transform,
    build_classifier_validation_transform,
)

SIZE = CLASSIFIER_IMAGE_SIZE
MEAN, STD = list(IMAGENET_MEAN), list(IMAGENET_STD)
ClassifierTransform = Callable[[Image.Image], torch.Tensor]


class ClassifierDataConfig(Protocol):
    """분류 DataLoader factory가 사용하는 최소 설정."""

    batch_size: int
    seed: int
    class_map_path: Path
    data_summary_path: Path


def train_transform() -> ClassifierTransform:
    """명세서 6.2: crop 후 padding 전에 회전. 새로 생긴 영역도 (124, 116, 104)."""
    return build_classifier_train_transform()


def val_transform() -> ClassifierTransform:
    """명세서 6.3: 무작위 처리 없음."""
    return build_classifier_validation_transform()


def detection_transform() -> ClassifierTransform:
    """명세서 8.1: crop_pill로 자른 탐지 결과 → float32 [3, 224, 224]. crop_pill이 이미 정사각형으로 만든다."""
    return build_classifier_inference_transform()


# ── 샘플 ─────────────────────────────────────────────────────────────────
META_KEYS = [
    "sample_id",
    "image_id",
    "image_path",
    "category_id",
    "item_seq",
    "bbox_xywh",
    "group_id",
    "drug_dir",
]


class PillCropDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: pd.DataFrame,
        root: str | Path,
        transform: ClassifierTransform,
    ) -> None:
        self.root, self.transform = Path(root), transform
        self.paths = samples.crop_path.tolist()
        self.targets = samples.class_index.tolist()
        self.metas = [
            {key: row[key] for key in META_KEYS} for row in samples.to_dict("records")
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with Image.open(self.root / self.paths[index]) as source_image:
            image = self.transform(source_image.convert("RGB"))
        return {
            "image": image,
            "target": self.targets[index],
            "meta": dict(self.metas[index]),
        }


def collate_classifier_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([s["image"] for s in samples]),
        "targets": torch.tensor([s["target"] for s in samples], dtype=torch.int64),
        "meta": [s["meta"] for s in samples],
    }


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ── 데이터 구성 함수 (명세서 2.2) ─────────────────────────────────────────
def build_classifier_inputs(
    *,
    config: ClassifierDataConfig,
    device: torch.device,
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    samples, class_map, data_summary = load_classifier_package(
        class_map_path=config.class_map_path,
        data_summary_path=config.data_summary_path,
    )
    root = Path(config.class_map_path).parent

    train_samples = samples[samples.split == "train"]
    val_samples = samples[samples.split == "val"]
    train_dataset = PillCropDataset(train_samples, root, train_transform())
    val_dataset = PillCropDataset(val_samples, root, val_transform())

    # 명세서 7절: 클래스 빈도의 역수로 뽑는다. sampler를 쓰므로 shuffle=False
    counts = data_summary["train_class_counts"]
    weights = torch.tensor(
        [1.0 / counts[str(target)] for target in train_dataset.targets],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    uses_cuda = (
        torch.device(device).type == "cuda"
    )  # 명세서 2.3: CUDA는 worker 2·pin_memory, MPS/CPU는 0·False
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": 2 if uses_cuda else 0,
        "pin_memory": uses_cuda,
        "collate_fn": collate_classifier_samples,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=False,
        generator=torch.Generator().manual_seed(config.seed),
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    return train_loader, val_loader, class_map, data_summary


if __name__ == "__main__":
    from classifier.training_data import validate_classifier_training_data

    if len(sys.argv) != 2:
        sys.exit("사용: python -m data_loading.classifier <class_map.json 경로>")
    path = Path(sys.argv[1])
    cfg = SimpleNamespace(
        batch_size=32,
        seed=42,
        class_map_path=path,
        data_summary_path=path.with_name("data_summary.json"),
    )
    loaders = build_classifier_inputs(config=cfg, device=torch.device("cpu"))
    validate_classifier_training_data(
        loaders,
        expected_num_classes=loaders[2]["num_classes"],
    )
    print(
        "입력 검증 완료", loaders[3]["num_train_samples"], loaders[3]["num_val_samples"]
    )
