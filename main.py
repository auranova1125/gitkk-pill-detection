"""YAML 설정으로 학습, 평가, 통합 추론과 smoke test를 실행하는 CLI."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def build_parser() -> argparse.ArgumentParser:
    """지원 명령과 공통 옵션을 정의한 argument parser를 만든다."""

    parser = argparse.ArgumentParser(
        description="경구약제 분류기·탐지기 실행 진입점",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    classifier_parser = subparsers.add_parser(
        "train-classifier",
        help="분류기 학습을 실행한다.",
    )
    classifier_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "classifier.yaml",
    )

    detector_parser = subparsers.add_parser(
        "train-detector",
        help="탐지기 학습을 실행한다.",
    )
    detector_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "detector.yaml",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="분류기 또는 탐지기 checkpoint를 평가한다.",
    )
    evaluate_parser.add_argument("target", choices=("classifier", "detector"))
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path)

    inference_parser = subparsers.add_parser(
        "infer",
        help="탐지기와 분류기의 통합 추론을 실행한다.",
    )
    inference_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "pipeline.yaml",
    )

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="합성 데이터로 공통 전처리와 학습 1 step을 확인한다.",
    )
    smoke_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 인자를 해석해 해당 workflow에 전달한다."""

    args = build_parser().parse_args(argv)
    if args.command == "train-classifier":
        from configs.schema import load_classifier_experiment_config
        from pipeline.workflows import train_classifier_experiment

        experiment = load_classifier_experiment_config(args.config)
        result = train_classifier_experiment(experiment)
        _print_json(
            {
                "run_dir": str(result.run_dir),
                "best_checkpoint_path": str(result.best_checkpoint_path),
                "completed_epoch": result.completed_epoch,
                "best_epoch": result.best_epoch,
            }
        )
        return 0

    if args.command == "train-detector":
        from configs.schema import load_detector_experiment_config
        from pipeline.workflows import train_detector_experiment

        experiment = load_detector_experiment_config(args.config)
        result = train_detector_experiment(experiment)
        _print_json(
            {
                "run_dir": str(result.run_dir),
                "best_checkpoint_path": str(result.best_checkpoint_path),
                "completed_epoch": result.completed_epoch,
                "best_epoch": result.best_epoch,
            }
        )
        return 0

    if args.command == "evaluate":
        from configs.schema import (
            load_classifier_experiment_config,
            load_detector_experiment_config,
        )
        from pipeline.workflows import (
            evaluate_classifier_experiment,
            evaluate_detector_experiment,
        )

        if args.target == "classifier":
            experiment = load_classifier_experiment_config(args.config)
            artifacts = evaluate_classifier_experiment(
                experiment,
                args.checkpoint,
                output_dir=args.output_dir,
            )
            _print_json(artifacts.metrics.to_dict())
        else:
            experiment = load_detector_experiment_config(args.config)
            result = evaluate_detector_experiment(
                experiment,
                args.checkpoint,
                output_dir=args.output_dir,
            )
            _print_json(result.to_dict())
        return 0

    if args.command == "infer":
        from configs.schema import load_pipeline_experiment_config
        from pipeline.workflows import run_pipeline_experiment

        experiment = load_pipeline_experiment_config(args.config)
        result = run_pipeline_experiment(experiment)
        _print_json(
            {
                "output_dir": str(experiment.output_dir),
                "num_predictions": len(result.predictions),
                "num_detector_errors": len(result.detector_errors),
                "num_classifier_errors": len(result.classifier_errors),
            }
        )
        return 0

    if args.command == "smoke":
        from pilldata.synthetic import run_synthetic_smoke

        _print_json(run_synthetic_smoke(device_name=args.device).to_dict())
        return 0

    raise AssertionError(f"unsupported command: {args.command}")


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
