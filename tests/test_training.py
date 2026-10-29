"""분류기 학습 및 평가 함수 테스트."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from classifier.evaluate import compute_classification_metrics, evaluate_model
from classifier.model import TrainingStage, build_model
from classifier.train import (
    build_grad_scaler,
    build_optimizer,
    build_scheduler,
    resolve_training_stage,
    select_device,
    train_one_epoch,
)
from configs.classifier import ClassifierConfig
from utils.runtime import capture_rng_state, restore_rng_state, set_seed

NUM_CLASSES = 8


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.images = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ],
            dtype=torch.float32,
        )
        self.targets = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"images": self.images[index], "targets": self.targets[index]}


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(4, num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


@pytest.fixture
def train_config(tmp_path: Path) -> ClassifierConfig:
    return ClassifierConfig(
        run_name="training-test",
        output_dir=tmp_path / "outputs",
        class_map_path=tmp_path / "class_map.json",
        data_summary_path=tmp_path / "data_summary.json",
    )


def test_set_seed_reproduces_python_numpy_and_torch() -> None:
    set_seed(42)
    first_values = (random.random(), np.random.rand(), torch.rand(1))

    set_seed(42)
    second_values = (random.random(), np.random.rand(), torch.rand(1))

    assert first_values[0] == second_values[0]
    assert first_values[1] == second_values[1]
    assert torch.equal(first_values[2], second_values[2])


def test_restore_rng_state_moves_device_tensors_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeviceState:
        def __init__(self, value: torch.Tensor) -> None:
            self.value = value
            self.cpu_called = False

        def cpu(self) -> torch.Tensor:
            self.cpu_called = True
            return self.value

    original_state = capture_rng_state()
    torch_state = DeviceState(original_state["torch"])
    cuda_state = DeviceState(original_state["torch"].clone())
    mps_state = DeviceState(original_state["torch"].clone())
    restored: dict[str, Any] = {}

    monkeypatch.setattr(
        torch, "set_rng_state", lambda value: restored.update(torch=value)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda value: restored.update(cuda=value),
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.mps,
        "set_rng_state",
        lambda value: restored.update(mps=value),
    )

    restore_rng_state(
        {
            **original_state,
            "torch": torch_state,
            "cuda": [cuda_state],
            "mps": mps_state,
        }
    )

    assert torch_state.cpu_called
    assert cuda_state.cpu_called
    assert mps_state.cpu_called
    assert restored["torch"].device.type == "cpu"
    assert restored["cuda"][0].device.type == "cpu"
    assert restored["mps"].device.type == "cpu"


def test_select_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert select_device() == torch.device("cuda")


def test_select_device_uses_mps_before_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert select_device() == torch.device("mps")


def test_select_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert select_device() == torch.device("cpu")


@pytest.mark.parametrize(
    ("epoch", "expected_stage"),
    [
        (0, TrainingStage.HEAD_ONLY),
        (2, TrainingStage.HEAD_ONLY),
        (3, TrainingStage.LAYER4_FINE_TUNING),
        (29, TrainingStage.LAYER4_FINE_TUNING),
    ],
)
def test_resolve_training_stage(
    epoch: int,
    expected_stage: TrainingStage,
) -> None:
    assert resolve_training_stage(epoch, head_only_epochs=3) is expected_stage


def test_build_head_optimizer(train_config: ClassifierConfig) -> None:
    model = build_model(NUM_CLASSES, pretrained=False)

    optimizer = build_optimizer(model, train_config, TrainingStage.HEAD_ONLY)

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["name"] == "fc"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert build_scheduler(optimizer, train_config, TrainingStage.HEAD_ONLY) is None


def test_build_fine_tuning_optimizer_and_scheduler(
    train_config: ClassifierConfig,
) -> None:
    model = build_model(NUM_CLASSES, pretrained=False)

    optimizer = build_optimizer(
        model,
        train_config,
        TrainingStage.LAYER4_FINE_TUNING,
    )
    scheduler = build_scheduler(
        optimizer,
        train_config,
        TrainingStage.LAYER4_FINE_TUNING,
    )

    assert [group["name"] for group in optimizer.param_groups] == ["layer4", "fc"]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-4, 5e-4]
    )
    trainable_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    optimized_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(optimized_parameter_ids) == len(set(optimized_parameter_ids))
    assert set(optimized_parameter_ids) == trainable_parameter_ids
    assert scheduler is not None
    assert scheduler.T_max == 27


def test_build_grad_scaler_is_disabled_for_cpu() -> None:
    scaler = build_grad_scaler(torch.device("cpu"))

    assert not scaler.is_enabled()


def test_train_one_epoch_updates_model() -> None:
    set_seed(42)
    model = TinyClassifier()
    data_loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    initial_weight = model.layers[1].weight.detach().clone()

    result = train_one_epoch(
        model,
        data_loader,
        criterion,
        optimizer,
        torch.device("cpu"),
    )

    assert result.num_samples == 4
    assert result.loss > 0
    assert 0 <= result.top1_accuracy <= 1
    assert result.top5_accuracy == 1
    assert not torch.equal(initial_weight, model.layers[1].weight)


def test_train_one_epoch_shows_progress(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = TinyClassifier()
    data_loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)

    train_one_epoch(
        model,
        data_loader,
        nn.CrossEntropyLoss(),
        torch.optim.AdamW(model.parameters(), lr=0.1),
        torch.device("cpu"),
        show_progress=True,
        progress_description="Classifier progress test",
    )

    assert "Classifier progress test" in capsys.readouterr().err


def test_compute_classification_metrics_excludes_unsupported_classes() -> None:
    logits = torch.tensor(
        [
            [9.0, 1.0, 0.0, -1.0, -2.0],
            [0.0, 9.0, 1.0, -1.0, -2.0],
            [9.0, 0.0, 1.0, -1.0, -2.0],
            [0.0, 9.0, 1.0, -1.0, -2.0],
        ]
    )
    targets = torch.tensor([0, 1, 2, 2], dtype=torch.int64)

    result = compute_classification_metrics(
        logits,
        targets,
        num_classes=5,
        loss=0.5,
    )

    assert result.loss == pytest.approx(0.5)
    assert result.top1_accuracy == pytest.approx(0.5)
    assert result.top5_accuracy == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(4 / 9)
    assert result.validation_class_coverage == pytest.approx(3 / 5)
    assert result.per_class[2].support == 2
    assert result.per_class[2].f1 == pytest.approx(0.0)
    assert result.per_class[3].f1 is None
    assert result.confusion_pairs[0].true_class_index == 2


def test_evaluate_model_aggregates_all_batches() -> None:
    set_seed(42)
    model = TinyClassifier()
    data_loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)

    result = evaluate_model(
        model,
        data_loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        num_classes=2,
    )

    assert result.num_samples == 4
    assert result.loss > 0
    assert 0 <= result.top1_accuracy <= 1
    assert result.top5_accuracy == 1
    assert result.validation_class_coverage == 1


@pytest.mark.parametrize(
    ("function_name", "kwargs"),
    [
        ("train", {}),
        ("evaluate", {"num_classes": 2}),
    ],
)
def test_epoch_functions_reject_empty_loader(
    function_name: str,
    kwargs: dict[str, Any],
) -> None:
    model = TinyClassifier()
    criterion = nn.CrossEntropyLoss()
    empty_loader: list[dict[str, torch.Tensor]] = []

    with pytest.raises(ValueError, match="at least one sample"):
        if function_name == "train":
            train_one_epoch(
                model,
                empty_loader,
                criterion,
                torch.optim.AdamW(model.parameters()),
                torch.device("cpu"),
            )
        else:
            evaluate_model(
                model,
                empty_loader,
                criterion,
                torch.device("cpu"),
                **kwargs,
            )
