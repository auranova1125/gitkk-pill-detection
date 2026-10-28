"""외부 패키지 없이 라벨·분할·경로를 검사하고 고정 분할의 표본을 고른다."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .settings import resolve_data_paths, resolve_image_path

SPLITS = ("train", "validation")


def read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def read_package(settings):
    package, roots = resolve_data_paths(settings)
    paths = [package / "split_manifest.csv"] + [
        package / "annotations" / f"{split}.json" for split in SPLITS
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "필요한 데이터 파일이 없습니다:\n" + "\n".join(missing)
        )
    with paths[0].open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    labels = {split: read_json(paths[i + 1]) for i, split in enumerate(SPLITS)}
    return package, roots, rows, labels


def image_location(image, roots):
    return resolve_image_path(image["file_name"], roots)


def inspect_package(settings, *, check_image_files=True):
    package, roots, rows, labels = read_package(settings)
    manifest = {}
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in manifest:
            raise ValueError(f"분할 목록 이미지 ID 중복: {image_id}")
        if row["split"] not in SPLITS:
            raise ValueError(f"알 수 없는 분할: {row['split']}")
        manifest[image_id] = row
    image_ids, annotation_ids = set(), set()
    groups = {}
    counts = {}
    for split in SPLITS:
        payload = labels[split]
        images = {}
        for image in payload["images"]:
            image_id = image["id"]
            if type(image_id) is not int or image_id < 0 or image_id in image_ids:
                raise ValueError(f"이미지 ID 오류 또는 중복: {image_id}")
            image_ids.add(image_id)
            images[image_id] = image
            if any(type(image[k]) is not int or image[k] <= 0 for k in ("width", "height")):
                raise ValueError(f"이미지 크기 오류: {image_id}")
            row = manifest.get(image_id)
            if row is None or row["split"] != split:
                raise ValueError(f"분할 목록과 라벨 배정 불일치: {image_id}")
            if row["image_path"] != image["file_name"]:
                raise ValueError(f"분할 목록과 이미지 경로 불일치: {image_id}")
            portable = str(image["file_name"]).replace("\\", "/")
            if row["dataset"] != portable.split("/", 1)[0]:
                raise ValueError(f"이미지 출처 불일치: {image_id}")
            name = portable.rsplit("/", 1)[-1]
            start = name.find("K-")
            prefix = name[start:].split("_", 1)[0] if start >= 0 else ""
            codes = prefix.removeprefix("K-").split("-")
            if not codes or any(len(c) != 6 or not c.isdigit() for c in codes):
                raise ValueError(f"알약 조합 형식 오류: {name}")
            if row["group_id"] != "K-" + "-".join(sorted(codes)):
                raise ValueError(f"조합 그룹 불일치: {image_id}")
            path = image_location(image, roots)
            if check_image_files and not path.is_file():
                raise FileNotFoundError(f"이미지 파일이 없습니다: {path}")
        per_image = defaultdict(int)
        for annotation in payload["annotations"]:
            aid = annotation["id"]
            if type(aid) is not int or aid < 0 or aid in annotation_ids:
                raise ValueError(f"annotation ID 오류 또는 중복: {aid}")
            annotation_ids.add(aid)
            iid = annotation["image_id"]
            if iid not in images:
                raise ValueError(f"분할 이미지에 연결되지 않은 annotation: {aid}")
            box = annotation["bbox"]
            if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
                raise ValueError(f"bbox 수치 오류: {aid}")
            x, y, w, h = box
            im = images[iid]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > im["width"] or y + h > im["height"]:
                raise ValueError(f"bbox 크기 또는 이미지 경계 오류: {aid}")
            if annotation.get("ignore", 0) != 0 or annotation.get("iscrowd", 0) != 0:
                raise ValueError(f"ignore 또는 iscrowd 값 오류: {aid}")
            area = annotation.get("area")
            if isinstance(area, bool) or not isinstance(area, (int, float)) or not math.isfinite(area) or not math.isclose(area, w * h, rel_tol=1e-6, abs_tol=1e-4):
                raise ValueError(f"bbox 면적 불일치: {aid}")
            per_image[iid] += 1
        for iid in images:
            if per_image[iid] < 1 or per_image[iid] != int(manifest[iid]["annotation_count"]):
                raise ValueError(f"annotation 개수 불일치 또는 빈 이미지: {iid}")
        if not images:
            raise ValueError(f"이미지가 없는 분할입니다: {split}")
        groups[split] = {manifest[iid]["group_id"] for iid in images}
        counts[split] = {"images": len(images), "boxes": sum(per_image.values()), "groups": len(groups[split])}
    if set(manifest) != image_ids:
        raise ValueError("분할 목록과 라벨 이미지 구성이 다릅니다.")
    overlap = groups["train"] & groups["validation"]
    if overlap:
        raise ValueError(f"Train/Validation 조합 그룹 중복: {len(overlap)}개")
    return {"package_dir": str(package), "counts": counts, "group_overlap_count": 0,
            "image_files_checked": check_image_files}


def create_smoke_package(settings, destination, samples_per_split):
    """기존 배정과 모든 객체를 유지해 임시 표본 라벨을 만든다. 원본은 수정하지 않는다."""
    if samples_per_split < 1:
        raise ValueError("분할별 표본 수는 1 이상이어야 합니다.")
    inspect_package(settings, check_image_files=False)
    _, roots, rows, labels = read_package(settings)
    selected = set()
    destination = Path(destination)
    (destination / "annotations").mkdir(parents=True)
    sources = {}
    for split in SPLITS:
        candidates = sorted((r for r in rows if r["split"] == split), key=lambda r: int(r["image_id"]))
        chosen = []
        # 가능한 경우 두 출처를 먼저 한 장씩 선택한다.
        for dataset in sorted({r["dataset"] for r in candidates}):
            match = next(r for r in candidates if r["dataset"] == dataset)
            if len(chosen) < samples_per_split:
                chosen.append(match)
        used = {int(r["image_id"]) for r in chosen}
        for row in candidates:
            if len(chosen) >= samples_per_split:
                break
            if int(row["image_id"]) not in used:
                chosen.append(row)
                used.add(int(row["image_id"]))
        ids = {int(r["image_id"]) for r in chosen}
        selected |= ids
        payload = dict(labels[split])
        payload["images"] = [im for im in payload["images"] if im["id"] in ids]
        payload["annotations"] = [a for a in payload["annotations"] if a["image_id"] in ids]
        for image in payload["images"]:
            path = image_location(image, roots)
            if not path.is_file():
                raise FileNotFoundError(f"소량 검사 이미지가 없습니다: {path}")
        (destination / "annotations" / f"{split}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        sources[split] = [{"image_id": int(r["image_id"]), "dataset": r["dataset"], "image_path": r["image_path"], "group_id": r["group_id"]} for r in chosen]
    with (destination / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(row for row in rows if int(row["image_id"]) in selected)
    return sources
