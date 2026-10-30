"""공통 runtime과 artifact I/O 테스트."""

import csv
import json
from pathlib import Path

import pytest
import torch

from utils.io import append_csv_row, read_json, write_json
from utils.runtime import build_grad_scaler, resolve_device, select_device


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected_device"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_select_device_priority(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected_device: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    assert select_device() == torch.device(expected_device)


@pytest.mark.parametrize("device_type", ["cpu", "mps"])
def test_grad_scaler_is_disabled_without_cuda(device_type: str) -> None:
    assert not build_grad_scaler(torch.device(device_type)).is_enabled()


def test_resolve_device_accepts_cpu_and_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_write_json_preserves_korean_and_adds_final_newline(tmp_path: Path) -> None:
    output_path = tmp_path / "artifact.json"

    write_json(output_path, {"메시지": "알약", "값": 1}, sort_keys=True)

    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "메시지": "알약",
        "값": 1,
    }
    assert read_json(output_path) == {"메시지": "알약", "값": 1}


def test_append_csv_row_writes_one_header_and_multiple_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.csv"

    append_csv_row(output_path, {"epoch": 0, "loss": 1.0})
    append_csv_row(output_path, {"epoch": 1, "loss": 0.5})

    with output_path.open(encoding="utf-8", newline="") as file:
        assert list(csv.DictReader(file)) == [
            {"epoch": "0", "loss": "1.0"},
            {"epoch": "1", "loss": "0.5"},
        ]
