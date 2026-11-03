"""기존 48장과 bbox 최종 판정을 반영해 탐지기 v2 패키지를 구성한다."""

import argparse
import copy
import csv
import filecmp
import hashlib
import json
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data_loading.detector_package import (
    calculate_detector_package_metadata,
    load_detector_package,
)
from utils.io import write_json


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def local_settings(package):
    original = json.loads((package / "data_settings.json").read_text())
    return {
        "package_dir": str(package.resolve()),
        "image_roots": {
            name: str(package.resolve()) for name in original["image_roots"]
        },
    }


def relative_image_path(value):
    """구형 출처/images 접두어만 제거하고 실제 패키지 상대 경로를 유지한다."""
    return re.sub(r"^(select|codeit)/images/", "", str(value).replace("\\", "/"))


def normalize_sources(source, output):
    """이미지를 복사하지 않고 JSON·manifest의 경로 계약만 정규화한다."""
    (output / "annotations").mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        labels = json.loads((source / "annotations" / f"{split}.json").read_text())
        for image in labels["images"]:
            image["file_name"] = relative_image_path(image["file_name"])
        write_json(output / "annotations" / f"{split}.json", labels)
    fields, rows = read_csv(source / "split_manifest.csv")
    for row in rows:
        row["image_path"] = relative_image_path(row["image_path"])
    write_csv(output / "split_manifest.csv", fields, rows)


def read_records(settings):
    """구형 v1은 임시 metadata로 검사하고, 현행 패키지는 strict 로더로 읽는다."""
    source = Path(settings["package_dir"])
    labels = json.loads((source / "annotations/train.json").read_text())
    legacy = any(
        relative_image_path(i["file_name"]) != i["file_name"] for i in labels["images"]
    )
    if not legacy:
        return load_detector_package(settings)
    with tempfile.TemporaryDirectory(prefix="detector-v1-metadata-") as temporary:
        normalized = Path(temporary)
        normalize_sources(source, normalized)
        temporary_settings = {**settings, "package_dir": str(normalized)}
        label_map, summary = calculate_detector_package_metadata(temporary_settings)
        write_json(normalized / "detector_label_map.json", label_map)
        write_json(normalized / "detector_data_summary.json", summary)
        return load_detector_package(temporary_settings)


def refresh_metadata(package):
    label_map, summary = calculate_detector_package_metadata(local_settings(package))
    write_json(package / "detector_label_map.json", label_map)
    write_json(package / "detector_data_summary.json", summary)
    return summary


def build(source, output, actions_path, excluded_path, report_path, resume=False):
    source, output = source.resolve(), output.resolve()
    if (
        (
            output.exists()
            and not (resume and (output / "BUILD_INCOMPLETE.txt").is_file())
        )
        or excluded_path.exists()
        or report_path.exists()
    ):
        raise FileExistsError(
            "출력 패키지·제외 기록·보고서는 기존 파일을 덮어쓰지 않습니다."
        )
    before, _, before_summary = read_records(local_settings(source))
    _, actions = read_csv(actions_path)
    names = [row["file_name"] for row in actions]
    if len(names) != 48 or len(set(names)) != 48:
        raise ValueError("기존 제외 목록은 중복 없는 48장이어야 합니다.")
    action_map = {row["file_name"]: row for row in actions}
    fields, manifest = read_csv(source / "split_manifest.csv")
    matches = [r for r in manifest if Path(r["file_name"]).name in action_map]
    if len(matches) != 48:
        raise ValueError("v1 manifest에서 제외 대상 48장을 모두 찾지 못했습니다.")
    for row in matches:
        if row["split"] != action_map[Path(row["file_name"]).name]["탐지기_split"]:
            raise ValueError("제외 목록과 v1의 split이 다릅니다.")
    drop_ids = {int(row["image_id"]) for row in matches}
    labels = {}
    for split in ("train", "validation"):
        original = json.loads((source / "annotations" / f"{split}.json").read_text())
        filtered = copy.deepcopy(original)
        filtered["images"] = [i for i in original["images"] if i["id"] not in drop_ids]
        filtered["annotations"] = [
            a for a in original["annotations"] if a["image_id"] not in drop_ids
        ]
        for image in filtered["images"]:
            image["file_name"] = relative_image_path(image["file_name"])
        labels[split] = filtered
    codes = {a["drug_code"] for label in labels.values() for a in label["annotations"]}
    original_codes = {
        c for rows in before.values() for r in rows for c in r["drug_codes"]
    }
    if codes != original_codes or len(codes) != 118:
        raise ValueError("제외 후 기존 118종이 유지되지 않습니다.")
    required_bytes = sum(
        Path(row["image_path"]).stat().st_size
        for rows in before.values()
        for row in rows
        if row["image_id"] not in drop_ids
    )
    existing_parent = output.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    free_bytes = shutil.disk_usage(existing_parent).free
    if free_bytes < required_bytes + 1024**3:
        raise OSError(
            f"저장 공간 부족: 이미지 {required_bytes:,} bytes + 여유 1GiB 필요, "
            f"현재 {free_bytes:,} bytes 사용 가능"
        )
    output.mkdir(parents=True, exist_ok=resume)
    incomplete = output / "BUILD_INCOMPLETE.txt"
    incomplete.write_text(
        "생성 중인 중간 패키지입니다. 이 파일이 있으면 학습·배포에 사용하지 마세요.\n",
        encoding="utf-8",
    )
    (output / "annotations").mkdir(exist_ok=resume)
    for split, label in labels.items():
        write_json(output / "annotations" / f"{split}.json", label)
    for row in manifest:
        row["image_path"] = relative_image_path(row["image_path"])
    write_csv(
        output / "split_manifest.csv",
        fields,
        [row for row in manifest if int(row["image_id"]) not in drop_ids],
    )
    original_settings = json.loads((source / "data_settings.json").read_text())
    settings = {key: original_settings[key] for key in ("package_dir", "image_roots")}
    colab_v2 = str(Path(settings["package_dir"]).parent / "v2")
    settings["package_dir"] = colab_v2
    settings["image_roots"] = {key: colab_v2 for key in settings["image_roots"]}
    write_json(output / "data_settings.json", settings)
    shutil.copy2(source / "detector_label_map.json", output / "detector_label_map.json")

    def copy_image(pair):
        src, dst = pair
        if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
            shutil.copy2(src, dst)
        if not filecmp.cmp(src, dst, shallow=False):
            raise ValueError(f"이미지 내용 복사 불일치: {dst}")

    for split, rows in before.items():
        directory = output / ("train" if split == "train" else "val") / "images"
        directory.mkdir(parents=True, exist_ok=resume)
        pairs = [
            (Path(row["image_path"]), directory / Path(row["image_path"]).name)
            for row in rows
            if row["image_id"] not in drop_ids
        ]
        expected_names = {dst.name for _, dst in pairs}
        if any(p.name not in expected_names for p in directory.iterdir()):
            raise ValueError(
                f"출력 이미지 폴더에 예상하지 못한 파일이 있습니다: {directory}"
            )
        with ThreadPoolExecutor(max_workers=8) as pool:
            for count, _ in enumerate(pool.map(copy_image, pairs), 1):
                if count % 250 == 0:
                    print(
                        f"{split}: {count}/{len(pairs)} 복사·내용 비교 완료", flush=True
                    )
        print(f"{split}: 이미지 복사 완료", flush=True)
    refresh_metadata(output)
    after, _, summary = read_records(local_settings(output))
    for split, rows in after.items():
        expected = [r for r in before[split] if r["image_id"] not in drop_ids]
        assert len(rows) == len(expected)
        for actual, old in zip(rows, expected, strict=True):
            for key in old:
                if key != "image_path" and actual[key] != old[key]:
                    raise ValueError(f"보존 데이터 변경: {key}")
            if (
                Path(actual["image_path"]).stat().st_size
                != Path(old["image_path"]).stat().st_size
            ):
                raise ValueError("이미지 복사 크기 불일치")
    exclusions = []
    for row in matches:
        action = action_map[Path(row["file_name"]).name]
        exclusions.append(
            {
                "file_name": Path(row["file_name"]).name,
                "split": row["split"],
                "annotation_count": row["annotation_count"],
                "source_recommendation": action["탐지기_권장_조치"],
                "applied_action": "이미지 및 bbox 전체 제외",
                "reason": action["메모"],
            }
        )
    excluded_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(excluded_path, list(exclusions[0]), exclusions)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 탐지기 데이터 v2 구성 보고서",
        "",
        "상태: 1단계 완료. 추가 bbox 자동 검사·육안 검수 전 중간 결과.",
        "",
        "기존 분할·ID·bbox를 유지하고 대회 이미지 3장을 포함한 총 48장의 이미지와 연결 bbox를 제외했다.",
        "",
        "| 분할 | v1 이미지 | 중간 v2 이미지 | v1 bbox | 중간 v2 bbox |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, name in (("train", "train"), ("validation", "val")):
        lines.append(
            f"| {split} | {len(before[split])} | {len(after[split])} | {before_summary[f'num_{name}_instances']} | {summary[f'num_{name}_instances']} |"
        )
    lines += [
        "",
        f"- v1 split_hash: `{before_summary['split_hash']}`",
        f"- 중간 v2 split_hash: `{summary['split_hash']}`",
        "- 기존 118종 유지, 로더 전체 검증 및 보존 기록 비교 통과.",
        "- Colab 설정 및 JSON의 출처별 경로 규칙 유지. 로컬 검증은 메모리에서만 경로 대체.",
        "- 육안 검수 미완료이므로 최종 ZIP 및 Drive 게시 전 추가 검수가 필요하다.",
        "",
        f"v1: `{source}`",
        f"\nv2 로컬: `{output}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    incomplete.unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def apply_reviews(package, reviews_path, excluded_path, report_path, audit_dir):
    """사용자 확정 판정의 오류 포함 사진 전체를 제외하고 중간본을 백업한다."""
    package = package.resolve()
    before, _, before_summary = read_records(local_settings(package))
    _, reviews = read_csv(reviews_path)
    if not reviews or len({r["sample_id"] for r in reviews}) != len(reviews):
        raise ValueError("판정 목록이 비어 있거나 sample_id가 중복됩니다.")
    if any(r["판정"] not in {"오류", "정상"} or not r["검수"] for r in reviews):
        raise ValueError("모든 후보에 사용자 최종 판정과 검수 기록이 필요합니다.")
    input_summary = json.loads((audit_dir / "input/input_summary.json").read_text())
    if before_summary["split_hash"] != input_summary["split_hash"]:
        raise ValueError("검수 대상 패키지와 현재 패키지의 split_hash가 다릅니다.")
    error_rows = [r for r in reviews if r["판정"] == "오류"]
    drop_names = {r["image_id"] for r in error_rows}
    drop_records = []
    keep_records = {}
    for split, rows in before.items():
        keep_records[split] = []
        for row in rows:
            original = re.sub(
                r"^(codeit_|source_\d+_)", "", Path(row["image_path"]).stem
            )
            if original in drop_names:
                related = [r for r in error_rows if r["image_id"] == original]
                if any(
                    r["split"] != split or r["조치"] != "이미지 제외" for r in related
                ):
                    raise ValueError("제외 조치 또는 split 불일치")
                drop_records.append((split, row, related))
            else:
                keep_records[split].append(row)
    if len(drop_records) != len(drop_names) or not drop_records:
        raise ValueError("제외 사진을 패키지에 일대일로 연결할 수 없습니다.")
    if (
        len(
            {c for rows in keep_records.values() for r in rows for c in r["drug_codes"]}
        )
        != 118
    ):
        raise ValueError("118종 유지 실패")
    fields, exclusions = read_csv(excluded_path)
    if {r["file_name"] for r in exclusions} & {
        Path(r["image_path"]).name for _, r, _ in drop_records
    }:
        raise ValueError("이미 제외한 이미지가 판정 대상에 있습니다.")
    backup = audit_dir / "stage1_backup"
    if backup.exists():
        raise FileExistsError(backup)
    backup.mkdir(parents=True)
    for relative in (
        "annotations/train.json",
        "annotations/validation.json",
        "split_manifest.csv",
        "detector_data_summary.json",
        "data_settings.json",
        "detector_label_map.json",
    ):
        dest = backup / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / relative, dest)
    shutil.copy2(excluded_path, backup / "excluded_images.csv")
    shutil.copy2(report_path, backup / "data_v2_report.md")
    marker = package / "BUILD_INCOMPLETE.txt"
    marker.write_text("육안 검수 결과 반영 중. 완료 전 사용 금지.\n", encoding="utf-8")
    drop_ids = {r["image_id"] for _, r, _ in drop_records}
    for split in ("train", "validation"):
        path = package / "annotations" / f"{split}.json"
        labels = json.loads(path.read_text())
        labels["images"] = [r for r in labels["images"] if r["id"] not in drop_ids]
        labels["annotations"] = [
            r for r in labels["annotations"] if r["image_id"] not in drop_ids
        ]
        write_json(path, labels)
    manifest_fields, manifest = read_csv(package / "split_manifest.csv")
    write_csv(
        package / "split_manifest.csv",
        manifest_fields,
        [r for r in manifest if int(r["image_id"]) not in drop_ids],
    )
    for split, row, related in drop_records:
        source_image = Path(row["image_path"])
        dest = (
            backup
            / ("train" if split == "train" else "val")
            / "images"
            / source_image.name
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        source_image.rename(dest)
        exclusions.append(
            {
                "file_name": source_image.name,
                "split": split,
                "annotation_count": len(row["boxes"]),
                "source_recommendation": "bbox 위치 검수 사용자 확정",
                "applied_action": "이미지 및 bbox 전체 제외",
                "reason": " / ".join(dict.fromkeys(r["메모"] for r in related)),
            }
        )
    refresh_metadata(package)
    after, _, summary = read_records(local_settings(package))
    if after != keep_records:
        raise ValueError("보존 이미지의 split·ID·bbox·메타데이터 변경 감지")
    for split, rows in after.items():
        folder = package / ("train" if split == "train" else "val") / "images"
        actual = {
            p.name
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        }
        if actual != {Path(r["image_path"]).name for r in rows}:
            raise ValueError("JSON과 실제 이미지 목록 불일치")
    build_metadata = dict(
        build_stage="human_review_applied_validated",
        excluded_images=len(exclusions),
        reviewed_candidates=len(reviews),
        reviews_sha256=hashlib.sha256(reviews_path.read_bytes()).hexdigest(),
    )
    write_csv(excluded_path, fields, exclusions)
    audit_summary_path = audit_dir / "output/audit_summary.json"
    audit_summary = json.loads(audit_summary_path.read_text())
    audit_summary.update(
        stage="human_review_applied_validated", final_split_hash=summary["split_hash"]
    )
    audit_summary.update(build_metadata)
    write_json(audit_summary_path, audit_summary)
    report = report_path.read_text()
    report = report.replace(
        "상태: 1단계 완료. 추가 bbox 자동 검사·육안 검수 전 중간 결과.",
        "상태: 기존 48장 및 추가 검수 102장 제외 반영·로더 검증 완료. Drive 업로드 전.",
    )
    report += "\n## 최종 v2 반영 및 검증 (2026-11-03)\n\n"
    report += f"기존 48장 + 추가 {len(drop_records)}장 = 총 {len(exclusions)}장 제외. 해당 사진의 모든 bbox를 제거했다.\n\n"
    report += "| 분할 | 최종 이미지 | 최종 bbox |\n|---|---:|---:|\n"
    for split, name in (("train", "train"), ("validation", "val")):
        report += (
            f"| {split} | {len(after[split])} | {summary[f'num_{name}_instances']} |\n"
        )
    report += f"\n최종 split_hash: `{summary['split_hash']}`\n\n"
    report += "118종 유지, JSON·manifest·실제 이미지 일치, 제외 대상 부재, 보존 기록 동일성 및 로더 검증 통과.\n"
    report += "Colab 설정과 JSON 경로 규칙을 유지했다. Drive v1은 변경하지 않았다.\n"
    report += "자동 후보에 대한 육안 검수이며 6,302장 전체의 육안 전수 검수를 의미하지 않는다.\n"
    report += f"중간본 메타데이터와 추가 제외 이미지는 `{backup}`에 보존했다.\n"
    report_path.write_text(report, encoding="utf-8")
    marker.unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def normalize_package(package, audit_dir):
    """구형 v2의 이미지·판정을 유지하며 현행 경로·summary로 변환한다."""
    before, _, _ = read_records(local_settings(package))
    backup = audit_dir / "pre_main_metadata"
    if backup.exists():
        raise FileExistsError(backup)
    backup.mkdir(parents=True)
    for relative in (
        "annotations/train.json",
        "annotations/validation.json",
        "split_manifest.csv",
        "data_settings.json",
        "detector_label_map.json",
        "detector_data_summary.json",
    ):
        dest = backup / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / relative, dest)
    marker = package / "BUILD_INCOMPLETE.txt"
    marker.write_text(
        "현행 metadata 계약으로 변환 중. 완료 전 사용 금지.\n", encoding="utf-8"
    )
    normalize_sources(package, package)
    settings_path = package / "data_settings.json"
    original = json.loads(settings_path.read_text())
    write_json(
        settings_path, {key: original[key] for key in ("package_dir", "image_roots")}
    )
    summary = refresh_metadata(package)
    after, _, _ = load_detector_package(local_settings(package))
    if before != after:
        raise ValueError("경로 정규화 중 보존 이미지의 기록 변경 감지")
    summary_path = audit_dir / "output/audit_summary.json"
    audit = json.loads(summary_path.read_text())
    audit["previous_format_final_split_hash"] = audit["final_split_hash"]
    audit["final_split_hash"] = summary["split_hash"]
    audit["metadata_contract"] = "current_main_relative_paths_and_exact_summary"
    write_json(summary_path, audit)
    marker.unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--apply-reviews", action="store_true")
    parser.add_argument("--normalize-package", action="store_true")
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("etl/detector/label_audit/position_suspects.csv"),
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("datasets/select/position_audit")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/select/v2"))
    parser.add_argument(
        "--resume", action="store_true", help="미완성 표시가 있는 출력만 재개"
    )
    parser.add_argument(
        "--actions",
        type=Path,
        default=Path("etl/detector/label_audit/detector_actions.csv"),
    )
    parser.add_argument(
        "--excluded",
        type=Path,
        default=Path("etl/detector/label_audit/excluded_images.csv"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/detector/data_v2_report.md")
    )
    args = parser.parse_args()
    if args.normalize_package:
        normalize_package(args.output_dir, args.audit_dir)
        return
    if args.apply_reviews:
        apply_reviews(
            args.output_dir, args.reviews, args.excluded, args.report, args.audit_dir
        )
        return
    if args.source_dir is None:
        parser.error("1단계 생성에는 --source-dir이 필요합니다.")
    build(
        args.source_dir,
        args.output_dir,
        args.actions,
        args.excluded,
        args.report,
        args.resume,
    )


if __name__ == "__main__":
    main()
