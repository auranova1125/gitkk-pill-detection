"""정답 annotation 없이 원본 이미지를 탐지·분류 통합 추론 입력으로 만든다."""

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class PillRecognitionDataConfig(Protocol):
    """통합 추론 DataLoader가 사용하는 최소 설정."""

    input_dir: Path


class PillRecognitionImageDataset(Dataset[dict[str, Any]]):
    """폴더 안의 이미지를 정렬된 순서로 읽는 추론 Dataset."""

    def __init__(self, input_dir: Path) -> None:
        self.input_dir = Path(input_dir).expanduser().resolve()
        if not self.input_dir.is_dir():
            raise FileNotFoundError(
                f"pill recognition input directory does not exist: {self.input_dir}"
            )
        self.image_paths = sorted(
            path
            for path in self.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise ValueError(
                f"pill recognition input directory contains no supported images: "
                f"{self.input_dir}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size
            image_tensor = (
                torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
                .permute(2, 0, 1)
                .contiguous()
                .to(torch.float32)
                .div_(255)
            )
        return {
            "image": image_tensor,
            "meta": {
                "image_id": index,
                "file_name": image_path.relative_to(self.input_dir).as_posix(),
                "image_path": str(image_path),
                "original_size": [height, width],
            },
        }


def collate_pill_recognition_images(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """크기가 다른 원본 이미지를 list batch로 묶는다."""

    return {
        "images": [sample["image"] for sample in samples],
        "meta": [sample["meta"] for sample in samples],
    }


def build_pill_recognition_inputs(
    *,
    config: PillRecognitionDataConfig,
    batch_size: int,
) -> DataLoader[dict[str, Any]]:
    """통합 추론 이미지 폴더에서 DataLoader를 만든다."""

    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    dataset = PillRecognitionImageDataset(config.input_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pill_recognition_images,
    )
