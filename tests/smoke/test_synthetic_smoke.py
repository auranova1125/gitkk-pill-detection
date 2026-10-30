"""공통 합성 데이터로 분류·탐지 학습 1 step을 검사한다."""

import json

import pytest

from main import main


@pytest.mark.smoke
def test_main_runs_cpu_synthetic_smoke(capsys) -> None:
    assert main(["smoke", "--device", "cpu"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["device"] == "cpu"
    assert result["classifier_crop_shape"] == [3, 224, 224]
    assert result["classifier_loss"] > 0.0
    assert result["detector_loss"] > 0.0
