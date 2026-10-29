"""설치와 실행 환경을 빠르게 확인하는 소형 합성 smoke test."""

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from classifier.train import train_one_epoch as train_classifier_epoch
from detector.train import train_one_epoch as train_detector_epoch
from pilldata.transforms import crop_tensor_xyxy, preprocess_classifier_crop
from utils.runtime import build_grad_scaler, resolve_device, set_seed


@dataclass(frozen=True, slots=True)
class SyntheticSmokeResult:
    """합성 smoke test에서 확인한 장치와 최소 결과."""

    device: str
    classifier_loss: float
    detector_loss: float
    classifier_crop_shape: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["classifier_crop_shape"] = list(self.classifier_crop_shape)
        return values


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class _TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        image_mean = torch.stack([image.mean() for image in images]).mean()
        box_mean = torch.stack([target["boxes"].mean() for target in targets]).mean()
        return {
            "classification": (self.weight - image_mean).square(),
            "bbox_regression": self.weight.square() * box_mean / 100.0,
        }


def run_synthetic_smoke(*, device_name: str = "auto") -> SyntheticSmokeResult:
    """공통 전처리와 분류기·탐지기 학습 1 step을 합성 데이터로 실행한다."""

    device = resolve_device(device_name)
    set_seed(42)

    classifier_model = _TinyClassifier().to(device)
    classifier_optimizer = torch.optim.SGD(classifier_model.parameters(), lr=0.1)
    classifier_batch = {
        "images": torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ],
            dtype=torch.float32,
        ).expand(-1, 3, -1, -1),
        "targets": torch.tensor([0, 1], dtype=torch.int64),
    }
    classifier_result = train_classifier_epoch(
        classifier_model,
        [classifier_batch],
        nn.CrossEntropyLoss(),
        classifier_optimizer,
        device,
        scaler=build_grad_scaler(device),
    )

    detector_model = _TinyDetector().to(device)
    detector_optimizer = torch.optim.SGD(detector_model.parameters(), lr=0.1)
    detector_batch = {
        "images": [torch.full((3, 8, 8), 0.25, dtype=torch.float32)],
        "targets": [
            {
                "boxes": torch.tensor([[1.0, 1.0, 6.0, 6.0]]),
                "labels": torch.ones(1, dtype=torch.int64),
            }
        ],
    }
    detector_result = train_detector_epoch(
        detector_model,
        [detector_batch],
        detector_optimizer,
        device,
        scaler=build_grad_scaler(device),
    )

    source_image = torch.full((3, 16, 20), 0.5, dtype=torch.float32)
    classifier_crop = preprocess_classifier_crop(
        crop_tensor_xyxy(source_image, (4.0, 4.0, 16.0, 12.0))
    )
    return SyntheticSmokeResult(
        device=str(device),
        classifier_loss=classifier_result.loss,
        detector_loss=detector_result.loss,
        classifier_crop_shape=tuple(classifier_crop.shape),
    )
