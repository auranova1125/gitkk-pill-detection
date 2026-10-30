"""터미널과 노트북의 검사 명령을 연결한다."""
import argparse
import json
import sys
from datetime import datetime, timezone

from .settings import load_settings, repository_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="탐지 데이터 검사와 공유 표본 모델 연결 검사")
    parser.add_argument("command", choices=("check", "smoke", "smoke-check"))
    parser.add_argument("--package", default="data/smoke_test")
    parser.add_argument("--report")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    args = parser.parse_args(argv)
    destination = repository_path(args.report or f"outputs/detector_data/{args.command}_report.json")
    package = repository_path(args.package)
    if args.command != "check" and destination.is_relative_to(package):
        parser.error("검사 보고서는 표본 폴더 밖에 저장하세요.")
    result = {"command": args.command, "status": "failed",
              "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        if args.command == "check":
            from .data_records import read_records
            _, _, summary = read_records(load_settings())
            result.update(scope="full_labels_and_paths", data_summary=summary,
                          pixel_read_all_images=False)
        elif args.command == "smoke-check":
            from .smoke_test import verify_samples
            _, manifest, summary = verify_samples(package)
            result.update(scope="sample_integrity_only", package_hash=manifest["package_hash"],
                          sample_files_verified=True, data_summary=summary, model_executed=False)
        else:
            from .smoke_test import run_model_smoke
            result.update(run_model_smoke(package, device_name=args.device))
        result["status"] = "passed"
        print("[통과] 검사 완료")
        for split, prefix in (("Train", "train"), ("Validation", "val")):
            summary = result["data_summary"]
            print(f"{split}: {summary[f'num_{prefix}_images']}장 / {summary[f'num_{prefix}_instances']}개 박스")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AssertionError, ImportError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(f"[실패] {error}", file=sys.stderr)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"검사 보고서: {destination}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
