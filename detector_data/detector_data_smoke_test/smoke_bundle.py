"""검사 기록의 동일 표본을 공유하고 전체 데이터 없이 재실행한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from detector_data.preflight import SPLITS, image_location, inspect_package, read_json, read_package
from detector_data.settings import DEFAULT_SETTINGS, REPOSITORY_ROOT, load_settings, repository_path

MANIFEST_NAME = "smoke_package_manifest.json"
LAYOUT_VERSION = "pill-smoke-package-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def environment_versions():
    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "torchvision", "torchmetrics", "numpy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def code_hashes(root=REPOSITORY_ROOT):
    result = {}
    for folder in ("detector", "detector_data", "detector_data/detector_data_smoke_test"):
        directory = Path(root) / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"공유 저장소의 코드 폴더가 없습니다: {directory}")
        for path in sorted(directory.glob("*.py")):
            # 같은 Git 파일의 Windows/Mac 줄바꿈 차이는 코드 변경으로 취급하지 않는다.
            normalized = path.read_text(encoding="utf-8").encode("utf-8")
            result[path.relative_to(root).as_posix()] = hashlib.sha256(normalized).hexdigest()
    return result


def source_hashes(package):
    return {name: sha256(package / name) for name in (
        "split_manifest.csv", "annotations/train.json", "annotations/validation.json"
    )}


def inside(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"허용되지 않는 패키지 경로: {name}")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"패키지 폴더를 벗어난 경로: {name}")
    return result


def settings_for_bundle(package, augmentation):
    settings = deepcopy(DEFAULT_SETTINGS)
    settings["package_dir"] = str(Path(package).resolve())
    settings["image_roots"] = {
        "select": str(Path(package).resolve() / "images/select_aihub"),
        "codeit": str(Path(package).resolve() / "images/select_codeit"),
    }
    settings["augmentation"] = dict(augmentation)
    return settings


def package_hash(manifest):
    # 생성 환경·시간과 분리하여 입력 파일, 코드, 표본, 증강의 동일성을 식별한다.
    keys = ("version", "files_sha256", "code_sha256", "selected_image_ids", "augmentation", "batch_size", "seed")
    value = {key: manifest[key] for key in keys}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def export_bundle(*, settings, report_path, output_path, repository_root=REPOSITORY_ROOT):
    """새 표본을 고르지 않고 이미 통과한 검사 기록의 이미지·전체 객체를 보존한다."""
    report_path, output_path = Path(report_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"같은 이름의 공유 ZIP이 있습니다. 다른 --output 이름을 지정하세요: {output_path}")
    report = read_json(report_path)
    if report.get("status") != "passed" or report.get("command") != "smoke":
        raise ValueError("통과한 smoke 검사 기록을 --from-report로 지정해야 합니다.")
    if report.get("input_validation") != "passed":
        raise ValueError("입력 검증 통과 기록이 없습니다. 먼저 smoke 검사를 실행하세요.")
    selected = report.get("selected_images")
    if not isinstance(selected, dict) or set(selected) != set(SPLITS):
        raise ValueError("검사 기록에 Train/Validation 표본 목록이 없습니다.")
    summary = report["data_summary"]
    if report.get("batch_size") != 2:
        raise ValueError("이 공유 패키지는 배치 2 기준입니다. 배치 2의 smoke 검사 기록을 사용하세요.")
    if any(len(selected[split]) != 2 for split in SPLITS):
        raise ValueError("Train 2장·Validation 2장이 기록된 smoke 결과가 필요합니다.")
    augmentation = summary["train_augmentation"]
    if augmentation != DEFAULT_SETTINGS["augmentation"]:
        raise ValueError("공유용 기본 검사는 증강을 사용하지 않은 smoke 결과를 사용하세요.")
    inspect_package(settings, check_image_files=False)
    package, roots, rows, labels = read_package(settings)
    original_hashes = source_hashes(package)
    if report.get("source_file_sha256") != original_hashes:
        raise ValueError("검사 기록과 현재 라벨·분할 파일이 다릅니다. 현재 자료로 smoke 검사를 먼저 실행하세요.")
    source_code_hashes = code_hashes(repository_root)
    all_selected_ids = set()
    counts = {}
    with TemporaryDirectory(prefix="pill-smoke-export-") as temporary:
        bundle = Path(temporary) / "smoke_test"
        (bundle / "annotations").mkdir(parents=True)
        bundle_roots = {name: bundle / "images" / folder for name, folder in (
            ("select", "select_aihub"), ("codeit", "select_codeit")
        )}
        for split, short in (("train", "train"), ("validation", "val")):
            ids = {entry["image_id"] for entry in selected[split]}
            if len(ids) != 2 or all_selected_ids & ids:
                raise ValueError("검사 기록의 표본 이미지 ID가 중복됩니다.")
            all_selected_ids |= ids
            payload = dict(labels[split])
            payload["images"] = [image for image in payload["images"] if image["id"] in ids]
            payload["annotations"] = [a for a in payload["annotations"] if a["image_id"] in ids]
            if len(payload["images"]) != 2:
                raise ValueError(f"현재 {split} 라벨에서 검사 표본을 찾지 못했습니다.")
            source_rows = {int(row["image_id"]): row for row in rows if row["split"] == split}
            for entry in selected[split]:
                row = source_rows[entry["image_id"]]
                for key in ("dataset", "image_path", "group_id"):
                    if entry[key] != row[key]:
                        raise ValueError(f"검사 표본과 분할 기록 불일치: {entry['image_id']}")
            for image in payload["images"]:
                source = image_location(image, roots)
                destination = image_location(image, bundle_roots)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if sha256(source) != sha256(destination):
                    raise ValueError(f"이미지 복사 검증 실패: {image['id']}")
            counts[split] = {"images": len(payload["images"]), "boxes": len(payload["annotations"])}
            if counts[split]["boxes"] != summary[f"num_{short}_instances"]:
                raise ValueError(f"검사 기록과 표본 annotation 수가 다릅니다: {split}")
            used_categories = {a["category_id"] for a in payload["annotations"]}
            payload["categories"] = [c for c in payload.get("categories", []) if c["id"] in used_categories]
            json_write(bundle / "annotations" / f"{split}.json", payload)
        with (bundle / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(row for row in rows if int(row["image_id"]) in all_selected_ids)
        # 공유 기록에서 개인 경로와 원본 전체 데이터의 경로 정보는 제외한다.
        reference = {key: report[key] for key in (
            "command", "status", "device", "batch_size", "label_map", "data_summary",
            "input_validation", "first_batch_reproducibility", "model_step", "selected_images"
        ) if key in report}
        json_write(bundle / "reference_result.json", reference)
        (bundle / "README.md").write_text(
            "# 동일 표본 스모크 테스트\n\n"
            "통과한 검사 기록에서 가져온 Train 2장과 Validation 2장입니다.\n"
            "기존 이미지 ID·annotation ID·bbox·분할 배정을 유지합니다.\n"
            "전체 학습 데이터와 class_mapping.csv는 이 검사에 필요하지 않습니다.\n\n"
            "GitHub의 detector/, detector_data/, detector_data/detector_data_smoke_test/ 코드가 있는 저장소에서 실행하세요.\n"
            "압축을 풀어 datasets/smoke_test/에 배치합니다.\n\n"
            "Mac/Linux 터미널:\n\n```bash\nsource .venv/bin/activate\n"
            "python -m detector_data.detector_data_smoke_test.smoke_bundle check\n"
            "python -m detector_data.detector_data_smoke_test.smoke_bundle run --model-step\n```\n\n"
            "Windows에서는 먼저 .venv\\Scripts\\Activate.ps1로 환경을 활성화합니다.\n"
            "코드와 패키지의 해시를 검사한 뒤 같은 표본을 읽습니다.\n"
            "장치·라이브러리가 다르면 손실값은 달라질 수 있습니다.\n"
            "순전파·역전파 통과를 확인하며 50 epoch 학습과 checkpoint 저장은 하지 않습니다.\n",
            encoding="utf-8",
        )
        manifest = {
            "version": LAYOUT_VERSION,
            "counts": counts,
            "selected_image_ids": {split: [entry["image_id"] for entry in selected[split]] for split in SPLITS},
            "augmentation": dict(augmentation), "batch_size": 2, "seed": 42,
            "source_label_sha256": original_hashes,
            "image_hash_timing": "export_time",
            "code_hash_timing": "export_time",
            "reference_report_name": report_path.name,
            "export_environment": environment_versions(),
            "code_sha256": source_code_hashes,
            "files_sha256": {p.relative_to(bundle).as_posix(): sha256(p)
                             for p in sorted(bundle.rglob("*")) if p.is_file()},
        }
        manifest["package_hash"] = package_hash(manifest)
        json_write(bundle / MANIFEST_NAME, manifest)
        verify_bundle(bundle, repository_root=repository_root)
        if source_hashes(package) != original_hashes:
            raise ValueError("공유 ZIP 생성 중 원본 라벨이 변경되었습니다.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            archive = zipfile.ZipFile(output_path, "x", compression=zipfile.ZIP_DEFLATED)
        except FileExistsError:
            raise FileExistsError(f"공유 ZIP이 이미 있습니다: {output_path}") from None
        try:
            with archive:
                for path in sorted(bundle.rglob("*")):
                    if path.is_file():
                        archive.write(path, "smoke_test/" + path.relative_to(bundle).as_posix())
            with zipfile.ZipFile(output_path) as saved:
                if saved.testzip() is not None:
                    raise ValueError("공유 ZIP 무결성 검사 실패")
                for path in bundle.rglob("*"):
                    if path.is_file() and saved.read("smoke_test/" + path.relative_to(bundle).as_posix()) != path.read_bytes():
                        raise ValueError("공유 ZIP 저장 내용 불일치")
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
    return {"status": "passed", "output": str(output_path), "counts": counts,
            "package_hash": manifest["package_hash"], "source_files_unchanged": True}


def verify_bundle(package, *, repository_root=REPOSITORY_ROOT):
    package = Path(package).resolve()
    manifest = read_json(package / MANIFEST_NAME)
    if manifest["version"] != LAYOUT_VERSION or package_hash(manifest) != manifest["package_hash"]:
        raise ValueError("스모크 패키지 버전 또는 식별 해시가 일치하지 않습니다.")
    expected = set(manifest["files_sha256"])
    actual = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    if actual != expected | {MANIFEST_NAME}:
        raise ValueError("공유 패키지의 파일 구성이 기록과 다릅니다.")
    for name, expected_hash in manifest["files_sha256"].items():
        if sha256(inside(package, name)) != expected_hash:
            raise ValueError(f"공유 파일이 누락되거나 변경되었습니다: {name}")
    current_code = code_hashes(repository_root)
    if current_code != manifest["code_sha256"]:
        raise ValueError("공유자가 사용한 코드와 다릅니다. 같은 코드 버전을 받았는지 확인하세요.")
    settings = settings_for_bundle(package, manifest["augmentation"])
    inspected = inspect_package(settings)
    for split in SPLITS:
        if any(inspected["counts"][split][key] != manifest["counts"][split][key] for key in ("images", "boxes")):
            raise ValueError(f"표본 패키지 수량이 다릅니다: {split}")
    _, _, rows, _ = read_package(settings)
    for split in SPLITS:
        ids = {int(row["image_id"]) for row in rows if row["split"] == split}
        if ids != set(manifest["selected_image_ids"][split]):
            raise ValueError(f"표본 패키지 이미지 ID가 다릅니다: {split}")
    return manifest, settings


def main(argv=None):
    parser = argparse.ArgumentParser(description="동일 표본 스모크 테스트 자료 공유와 재실행")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="이미 통과한 검사 기록의 이미지 4장을 ZIP으로 공유")
    export.add_argument("--from-report", type=Path, default=Path("outputs/detector_data/model_smoke_report.json"))
    export.add_argument("--output", type=Path, default=Path("datasets/smoke_test.zip"))
    for name in ("check", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--package", type=Path, default=Path("datasets/smoke_test"))
        if name == "run":
            sub.add_argument("--model-step", action="store_true")
            sub.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
            sub.add_argument("--report", type=Path, default=Path("outputs/detector_data/shared_smoke_report.json"))
    args = parser.parse_args(argv)
    result = {"command": args.command, "status": "failed"}
    try:
        if args.command == "export":
            result.update(export_bundle(settings=load_settings(), report_path=repository_path(args.from_report),
                                        output_path=repository_path(args.output)))
            print("[통과] 동일 표본 공유 ZIP 생성 완료")
            print(f"공유할 파일: {result['output']}")
        else:
            package = repository_path(args.package)
            manifest, settings = verify_bundle(package)
            result.update(package_hash=manifest["package_hash"], package_verification="passed")
            if args.command == "run":
                from detector_data.__main__ import validate_loaders
                runtime_args = argparse.Namespace(command="smoke", batch_size=manifest["batch_size"],
                                                  device=args.device, model_step=args.model_step)
                result.update(validate_loaders(settings, runtime_args))
                verify_bundle(package)
                result.update(environment=environment_versions(), scope="shared_fixed_samples")
                print(f"모델 연결 검사: {result['model_step']['status']}")
            result["status"] = "passed"
            print("[통과] 공유 스모크 패키지 검사 완료")
        print(f"패키지 식별값: {result['package_hash']}")
        exit_code = 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AssertionError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(f"[실패] {error}", file=sys.stderr)
        exit_code = 1
    if args.command == "run":
        destination = repository_path(args.report)
        # 결과 JSON을 읽기 전용 공유 패키지 내부에 쓰지 않는다.
        if destination.is_relative_to(repository_path(args.package)):
            print("[실패] --report는 공유 패키지 바깥을 지정하세요.", file=sys.stderr)
            return 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        json_write(destination, result)
        print(f"검사 기록: {destination}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
