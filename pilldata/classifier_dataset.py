"""경구약제 분류기 Dataset과 DataLoader factory.

학습 노트북:
    data_factory_path = "pilldata.classifier_dataset:build_classifier_inputs"
    config.class_map_path    = "<training_package_classifier.zip을 푼 폴더>/class_map.json"
    config.data_summary_path = "<training_package_classifier.zip을 푼 폴더>/data_summary.json"

class_map.json과 같은 폴더에 samples.csv와 crops/가 있어야 한다.
crops/의 PNG는 원본 사진에서 bbox 사방 5% 여백을 더해 이미지 경계로 clip해 자른 직사각형이다 (명세서 6.1의 1~3).
이후 처리(학습: 회전 → 중앙 정사각형 padding, 공통: 224 resize → normalize)는 이 파일에서 한다.

탐지 결과를 분류할 때 (명세서 8절):
    from pilldata.classifier_dataset import detection_transform
    from pilldata.transforms import crop_pill
    image = detection_transform()(crop_pill(original_image, bbox_xywh))   # float32 [3, 224, 224]

입력 검증:  python -m pilldata.classifier_dataset <class_map.json 경로>
"""

import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from pilldata.transforms import (
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


def train_transform():
    """명세서 6.2: crop 후 padding 전에 회전. 새로 생긴 영역도 (124, 116, 104)."""
    return build_classifier_train_transform()


def val_transform():
    """명세서 6.3: 무작위 처리 없음."""
    return build_classifier_validation_transform()


def detection_transform():
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


def read_samples(root):
    df = pd.read_csv(Path(root) / "samples.csv", dtype=str, keep_default_na=False)
    df["class_index"] = df.class_index.astype(int)
    df["category_id"] = df.category_id.astype(int)
    df["bbox_xywh"] = df.bbox_xywh.map(lambda s: [float(v) for v in json.loads(s)])
    df["drug_dir"] = df.drug_dir.astype(object).where(
        df.drug_dir != "", None
    )  # 명세서 3.3: 정보가 없으면 None (NaN 아님)
    return df


class PillCropDataset(Dataset):
    def __init__(self, df, root, transform):
        self.root, self.transform = Path(root), transform
        self.paths = df.crop_path.tolist()
        self.targets = df.class_index.tolist()
        self.metas = [{k: r[k] for k in META_KEYS} for r in df.to_dict("records")]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            image = self.transform(im.convert("RGB"))
        return {"image": image, "target": self.targets[i], "meta": dict(self.metas[i])}


def collate_classifier_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([s["image"] for s in samples]),
        "targets": torch.tensor([s["target"] for s in samples], dtype=torch.int64),
        "meta": [s["meta"] for s in samples],
    }


def _seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ── data_summary ─────────────────────────────────────────────────────────
def split_hash(df):
    key = {
        "version": SPLIT_VERSION,
        "train": sorted(df[df.split == "train"].sample_id),
        "validation": sorted(df[df.split == "val"].sample_id),
    }
    return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()


def make_data_summary(df, num_classes):
    tr, va = df[df.split == "train"], df[df.split == "val"]

    def counts(d):
        c = d.class_index.value_counts()
        return {str(i): int(c.get(i, 0)) for i in range(num_classes)}

    vc = counts(va)
    return {
        "num_classes": num_classes,
        "num_train_samples": len(tr),
        "num_val_samples": len(va),
        "train_class_counts": counts(tr),
        "val_class_counts": vc,
        "train_group_count": tr.group_id.nunique(),
        "val_group_count": va.group_id.nunique(),
        "group_overlap_count": len(set(tr.group_id) & set(va.group_id)),
        "validation_class_indices": [i for i in range(num_classes) if vc[str(i)] > 0],
        "split_hash": split_hash(df),
    }


# ── 데이터 구성 함수 (명세서 2.2) ─────────────────────────────────────────
def build_classifier_inputs(*, config, device: torch.device):
    class_map = json.loads(Path(config.class_map_path).read_text(encoding="utf-8"))
    data_summary = json.loads(
        Path(config.data_summary_path).read_text(encoding="utf-8")
    )
    root = Path(config.class_map_path).parent
    df = read_samples(root)
    # 파일끼리 어긋나면(다른 분할의 data_summary 등) 학습 전에 멈춘다
    if make_data_summary(df, class_map["num_classes"]) != data_summary:
        raise ValueError(
            "data_summary.json이 samples.csv와 맞지 않습니다. 같은 패키지의 파일인지 확인하세요."
        )

    tr, va = df[df.split == "train"], df[df.split == "val"]
    train_ds = PillCropDataset(tr, root, train_transform())
    val_ds = PillCropDataset(va, root, val_transform())

    # 명세서 7절: 클래스 빈도의 역수로 뽑는다. sampler를 쓰므로 shuffle=False
    counts = data_summary["train_class_counts"]
    weights = torch.tensor(
        [1.0 / counts[str(c)] for c in train_ds.targets], dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_ds),
        replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    cuda = (
        torch.device(device).type == "cuda"
    )  # 명세서 2.3: CUDA는 worker 2·pin_memory, MPS/CPU는 0·False
    common = {
        "batch_size": config.batch_size,
        "num_workers": 2 if cuda else 0,
        "pin_memory": cuda,
        "collate_fn": collate_classifier_samples,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=False,
        generator=torch.Generator().manual_seed(config.seed),
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, class_map, data_summary


# ── 입력 검증 (명세서 9~10절과 같은 조건) ─────────────────────────────────
def check_inputs(
    train_loader, val_loader, class_map, data_summary, expected_num_classes=118
):
    n = expected_num_classes
    assert class_map["num_classes"] == n and len(class_map["classes"]) == n
    cls = {c["class_index"]: c for c in class_map["classes"]}
    assert sorted(cls) == list(range(n))
    assert (
        len({c["category_id"] for c in cls.values()}) == n
        and len({c["item_seq"] for c in cls.values()}) == n
    )
    assert all(
        isinstance(c["item_seq"], str) and c["item_seq"] and c["name"]
        for c in cls.values()
    )

    s = data_summary
    assert s["num_classes"] == n and s["group_overlap_count"] == 0
    assert (
        set(s["train_class_counts"])
        == set(s["val_class_counts"])
        == {str(i) for i in range(n)}
    )
    assert all(v >= 1 for v in s["train_class_counts"].values()), (
        "train에 없는 클래스가 있다"
    )
    assert s["num_train_samples"] == sum(s["train_class_counts"].values())
    assert s["num_val_samples"] == sum(s["val_class_counts"].values())
    assert s["validation_class_indices"] == [
        i for i in range(n) if s["val_class_counts"][str(i)] > 0
    ]
    assert s["split_hash"]

    for name, loader in (("train", train_loader), ("val", val_loader)):
        b = next(iter(loader))
        x, y, meta = b["images"], b["targets"], b["meta"]
        assert (
            x.dtype == torch.float32
            and x.ndim == 4
            and x.shape[1:] == (3, SIZE, SIZE)
            and x.shape[0] >= 1
        ), name
        assert torch.isfinite(x).all(), f"{name}: NaN/Inf"
        assert (
            y.dtype == torch.int64
            and y.shape == (x.shape[0],)
            and 0 <= y.min()
            and y.max() < n
        ), name
        assert (
            isinstance(meta, list)
            and len(meta) == x.shape[0]
            and all(isinstance(m, dict) for m in meta)
        ), name
        for t, m in zip(y.tolist(), meta):
            assert (
                m["category_id"] == cls[t]["category_id"]
                and m["item_seq"] == cls[t]["item_seq"]
            ), name
            assert all(
                m[k] for k in ("sample_id", "image_id", "image_path", "group_id")
            )
            b4 = m["bbox_xywh"]
            assert len(b4) == 4 and all(np.isfinite(b4)) and b4[2] > 0 and b4[3] > 0, (
                name
            )
            assert m["drug_dir"] is None or isinstance(m["drug_dir"], str), name
    v1, v2 = next(iter(val_loader))["images"], next(iter(val_loader))["images"]
    assert torch.equal(v1, v2), "val transform에 무작위 처리가 있다"
    assert train_loader.sampler.__class__.__name__ == "WeightedRandomSampler"
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("사용: python -m pilldata.classifier_dataset <class_map.json 경로>")
    path = Path(sys.argv[1])
    cfg = SimpleNamespace(
        batch_size=32,
        seed=42,
        class_map_path=path,
        data_summary_path=path.with_name("data_summary.json"),
    )
    loaders = build_classifier_inputs(config=cfg, device=torch.device("cpu"))
    check_inputs(*loaders)
    print(
        "입력 검증 완료", loaders[3]["num_train_samples"], loaders[3]["num_val_samples"]
    )
