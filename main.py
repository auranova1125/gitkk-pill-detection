"""YAML 설정으로 학습, 평가, 통합 추론과 smoke test를 실행하는 CLI."""

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
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
        "recognize-pills",
        help="탐지기와 분류기를 연결해 알약 품목을 인식한다.",
    )
    inference_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "pill_recognition.yaml",
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

    detector_data_parser = subparsers.add_parser(
        "check-detector-data",
        help="탐지 학습 패키지의 라벨·분할·경로를 검사한다.",
    )
    detector_data_parser.add_argument("--settings", type=Path)
    detector_data_parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/detector_data/check_report.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 인자를 해석해 해당 작업 함수에 전달한다."""

    args = build_parser().parse_args(argv)
    if args.command == "train-classifier":
        from classifier.train import train_classifier
        from configs.classifier import load_classifier_config

        config = load_classifier_config(args.config)
        result = train_classifier(config)
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
        from configs.detector import load_detector_config
        from detector.train import train_detector

        config = load_detector_config(args.config)
        result = train_detector(config)
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
        if args.target == "classifier":
            from classifier.evaluate import evaluate_classifier_checkpoint
            from configs.classifier import load_classifier_config

            config = load_classifier_config(args.config)
            artifacts = evaluate_classifier_checkpoint(
                config,
                args.checkpoint,
                output_dir=args.output_dir,
            )
            _print_json(artifacts.metrics.to_dict())
        else:
            from configs.detector import load_detector_config
            from detector.evaluate import evaluate_detector_checkpoint

            config = load_detector_config(args.config)
            result = evaluate_detector_checkpoint(
                config,
                args.checkpoint,
                output_dir=args.output_dir,
            )
            _print_json(result.to_dict())
        return 0

    if args.command == "recognize-pills":
        from configs.pill_recognition import load_pill_recognition_config
        from pipeline.pill_recognition import recognize_pills_from_config

        config = load_pill_recognition_config(args.config)
        result = recognize_pills_from_config(config)
        _print_json(
            {
                "output_dir": str(config.output_dir),
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

    if args.command == "check-detector-data":
        return _check_detector_data(args.settings, args.report)

    raise AssertionError(f"unsupported command: {args.command}")


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _check_detector_data(settings_path: Path | None, report_path: Path) -> int:
    """탐지 학습 패키지를 읽고 검사 결과를 JSON으로 저장한다."""
    from pilldata.detector_records import read_records
    from pilldata.detector_settings import load_detector_settings
    from utils.io import write_json

    result: dict[str, Any] = {
        "command": "check-detector-data",
        "status": "failed",
        "executed_at_utc": datetime.now(UTC).isoformat(),
    }
    try:
        _, _, summary = read_records(load_detector_settings(settings_path))
        result.update(
            status="passed",
            scope="full_labels_and_paths",
            data_summary=summary,
            pixel_read_all_images=False,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, result)
    _print_json(result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
