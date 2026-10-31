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

import hashlib
import json
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
SPLIT_VERSION = "detector-split-manifest-v1"  # 탐지기 split_manifest.csv의 그룹 분할을 그대로 따른 분할
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


def read_samples(root: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(root) / "samples.csv", dtype=str, keep_default_na=False)
    df["class_index"] = df.class_index.astype(int)
    df["category_id"] = df.category_id.astype(int)
    df["bbox_xywh"] = df.bbox_xywh.map(lambda s: [float(v) for v in json.loads(s)])
    df["drug_dir"] = df.drug_dir.astype(object).where(
        df.drug_dir != "", None
    )  # 명세서 3.3: 정보가 없으면 None (NaN 아님)
    return df


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


# ── data_summary ─────────────────────────────────────────────────────────
def split_hash(samples: pd.DataFrame) -> str:
    key = {
        "version": SPLIT_VERSION,
        "train": sorted(samples[samples.split == "train"].sample_id),
        "validation": sorted(samples[samples.split == "val"].sample_id),
    }
    return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()


def make_data_summary(
    samples: pd.DataFrame,
    num_classes: int,
) -> dict[str, Any]:
    train_samples = samples[samples.split == "train"]
    val_samples = samples[samples.split == "val"]

    def count_classes(split_samples: pd.DataFrame) -> dict[str, int]:
        class_counts = split_samples.class_index.value_counts()
        return {
            str(class_index): int(class_counts.get(class_index, 0))
            for class_index in range(num_classes)
        }

    val_class_counts = count_classes(val_samples)
    return {
        "num_classes": num_classes,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "train_class_counts": count_classes(train_samples),
        "val_class_counts": val_class_counts,
        "train_group_count": train_samples.group_id.nunique(),
        "val_group_count": val_samples.group_id.nunique(),
        "group_overlap_count": len(
            set(train_samples.group_id) & set(val_samples.group_id)
        ),
        "validation_class_indices": [
            class_index
            for class_index in range(num_classes)
            if val_class_counts[str(class_index)] > 0
        ],
        "split_hash": split_hash(samples),
    }


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
    class_map = json.loads(Path(config.class_map_path).read_text(encoding="utf-8"))
    data_summary = json.loads(
        Path(config.data_summary_path).read_text(encoding="utf-8")
    )
    root = Path(config.class_map_path).parent
    samples = read_samples(root)
    # 파일끼리 어긋나면(다른 분할의 data_summary 등) 학습 전에 멈춘다
    if make_data_summary(samples, class_map["num_classes"]) != data_summary:
        raise ValueError(
            "data_summary.json이 samples.csv와 맞지 않습니다. 같은 패키지의 파일인지 확인하세요."
        )

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
