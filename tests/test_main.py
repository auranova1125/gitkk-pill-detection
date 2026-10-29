"""CLI 명령 구성과 합성 smoke test."""

import json

from main import build_parser, main


def test_parser_exposes_expected_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["train-classifier"]).command == "train-classifier"
    assert parser.parse_args(["train-detector"]).command == "train-detector"
    assert (
        parser.parse_args(
            [
                "evaluate",
                "classifier",
                "--checkpoint",
                "best.pt",
                "--config",
                "configs/classifier.yaml",
            ]
        ).command
        == "evaluate"
    )
    assert parser.parse_args(["recognize-pills"]).command == "recognize-pills"
    assert parser.parse_args(["smoke", "--device", "cpu"]).command == "smoke"


def test_main_runs_cpu_synthetic_smoke(capsys) -> None:
    assert main(["smoke", "--device", "cpu"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["device"] == "cpu"
    assert result["classifier_crop_shape"] == [3, 224, 224]
    assert result["classifier_loss"] > 0.0
    assert result["detector_loss"] > 0.0
