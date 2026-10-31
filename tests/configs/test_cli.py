"""프로젝트 CLI 명령 구성 테스트."""

from main import build_parser


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
    assert parser.parse_args(["check-detector-data"]).command == "check-detector-data"
