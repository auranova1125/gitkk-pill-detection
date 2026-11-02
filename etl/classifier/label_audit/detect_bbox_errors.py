"""사진의 알약 영역과 annotation bbox를 비교해 위치 오류 후보를 만든다."""

import argparse
import io
import json
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

from etl.classifier.label_audit.review_records import preserve_human_reviews
from etl.classifier.paths import (
    COMBO_ZIPS,
    EXPORT,
    KAGGLE,
    LABEL_AUDIT_OUTPUT,
    LABEL_AUDIT_RECORDS,
    save_table,
    ts_zip,
)

DOWNSAMPLE_SCALE = 4
FOREGROUND_THRESHOLD = 12
MINIMUM_REGION_RATIO = 0.004
EDGE_REGION_RATIO = 0.02
RING_RATIO = 0.08


def background_difference(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """조명 그라데이션을 근사한 배경과 각 픽셀의 차이를 계산한다."""

    resized = cv2.resize(
        image,
        (
            image.shape[1] // DOWNSAMPLE_SCALE,
            image.shape[0] // DOWNSAMPLE_SCALE,
        ),
        interpolation=cv2.INTER_AREA,
    )
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    x_values = (grid_x / width - 0.5).ravel()
    y_values = (grid_y / height - 0.5).ravel()
    design = np.stack(
        [
            np.ones_like(x_values),
            x_values,
            y_values,
            x_values * x_values,
            y_values * y_values,
            x_values * y_values,
        ],
        axis=1,
    )
    pixels = lab.reshape(-1, 3)
    keep = np.ones(len(pixels), dtype=bool)
    residual = np.zeros_like(pixels)
    for _ in range(4):
        coefficients, *_ = np.linalg.lstsq(
            design[keep],
            pixels[keep],
            rcond=None,
        )
        residual = pixels - design @ coefficients
        distance = np.linalg.norm(residual, axis=1)
        keep = distance < max(3 * np.median(distance[keep]), 4)
    residual = residual.reshape(height, width, 3)
    lightness = residual[..., 0]
    chroma = np.hypot(residual[..., 1], residual[..., 2])
    difference = np.maximum.reduce(
        [
            chroma * 1.5,
            np.maximum(lightness, 0),
            np.maximum(-lightness - 25, 0),
        ]
    )
    return difference, resized


def find_pill_regions(image: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """알약으로 보이는 영역 mask와 원본 좌표계의 영역 bbox를 반환한다."""

    difference, resized = background_difference(image)
    foreground = (difference > FOREGROUND_THRESHOLD).astype(np.uint8)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 20, 50)
    foreground |= cv2.dilate(edges, np.ones((3, 3), np.uint8)) > 0
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
    )
    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    height, width = foreground.shape
    filled = np.zeros_like(foreground)
    regions: list[list[int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        on_edge = (
            x <= 1
            or y <= 1
            or x + box_width >= width - 1
            or y + box_height >= height - 1
        )
        if area < MINIMUM_REGION_RATIO * height * width or (
            on_edge and area < EDGE_REGION_RATIO * height * width
        ):
            continue
        cv2.drawContours(filled, [contour], -1, 1, -1)
        regions.append(
            [
                x * DOWNSAMPLE_SCALE,
                y * DOWNSAMPLE_SCALE,
                box_width * DOWNSAMPLE_SCALE,
                box_height * DOWNSAMPLE_SCALE,
            ]
        )
    return filled, regions


def bbox_iou(first: list[float], second: list[float]) -> float:
    """XYWH bbox 두 개의 IoU를 계산한다."""

    intersection_width = max(
        0.0,
        min(first[0] + first[2], second[0] + second[2]) - max(first[0], second[0]),
    )
    intersection_height = max(
        0.0,
        min(first[1] + first[3], second[1] + second[3]) - max(first[1], second[1]),
    )
    intersection = intersection_width * intersection_height
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def score_image(
    image: np.ndarray,
    boxes: list[list[float]],
) -> list[dict[str, int | float]]:
    """이미지의 각 annotation bbox에 위치 오류 점수를 계산한다."""

    filled, regions = find_pill_regions(image)
    matches: dict[int, float] = {}
    if regions:
        matrix = np.array(
            [[bbox_iou(box, region) for region in regions] for box in boxes]
        )
        for box_index, region_index in zip(
            *linear_sum_assignment(-matrix),
            strict=True,
        ):
            if matrix[box_index, region_index] > 0:
                matches[int(box_index)] = float(matrix[box_index, region_index])
    unmatched_regions = len(regions) - len(matches)

    scores: list[dict[str, int | float]] = []
    mask_height, mask_width = filled.shape
    for index, box in enumerate(boxes):
        x, y, width, height = [int(value / DOWNSAMPLE_SCALE) for value in box]
        x = min(max(x, 0), mask_width - 1)
        y = min(max(y, 0), mask_height - 1)
        x2 = min(mask_width, x + max(width, 1))
        y2 = min(mask_height, y + max(height, 1))
        margin = max(1, int(RING_RATIO * max(width, height)))
        ring = np.zeros(filled.shape, dtype=bool)
        ring[
            max(0, y - margin) : min(mask_height, y2 + margin),
            max(0, x - margin) : min(mask_width, x2 + margin),
        ] = True
        ring[y:y2, x:x2] = False
        for other_index, other_box in enumerate(boxes):
            if other_index == index:
                continue
            other_x, other_y, other_width, other_height = [
                int(value / DOWNSAMPLE_SCALE) for value in other_box
            ]
            ring[
                max(0, other_y) : min(mask_height, other_y + other_height),
                max(0, other_x) : min(mask_width, other_x + other_width),
            ] = False
        scores.append(
            {
                "fg_in": round(float(filled[y:y2, x:x2].mean()), 3),
                "ring_fg": round(
                    float(filled[ring].mean()) if ring.any() else 0.0,
                    3,
                ),
                "match_iou": round(matches.get(index, 0.0), 3),
                "n_boxes": len(boxes),
                "n_regions": len(regions),
                "unmatched_blobs": unmatched_regions,
            }
        )
    return scores


def position_priority(score: Any) -> int:
    """위치 점수를 0(정상)부터 3(경계)까지 우선순위로 변환한다."""

    if (
        score.fg_in < 0.15
        or 0 < score.match_iou < 0.5
        or (score.match_iou == 0 and (score.unmatched_blobs > 0 or score.fg_in < 0.5))
    ):
        return 1
    if score.match_iou == 0:
        return 2
    if score.match_iou < 0.6:
        return 3
    return 0


def build_source_image_index() -> dict[str, tuple[str, str]]:
    """AI Hub 원본 파일명을 zip 경로와 내부 경로에 연결한다."""

    index: dict[str, tuple[str, str]] = {}
    for number in COMBO_ZIPS:
        zip_path = ts_zip(number)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if member.endswith(".png"):
                    index[os.path.basename(member)] = (str(zip_path), member)
    return index


def load_source_image(
    image_id: str,
    source: str,
    image_index: dict[str, tuple[str, str]],
    archives: dict[str, zipfile.ZipFile],
) -> np.ndarray:
    """Kaggle 폴더 또는 AI Hub zip에서 RGB 이미지를 읽는다."""

    file_name = f"{image_id}.png"
    if source == "kaggle":
        with Image.open(KAGGLE / "train_images" / file_name) as image:
            return np.array(image.convert("RGB"))
    zip_path, member = image_index[file_name]
    archive = archives.setdefault(zip_path, zipfile.ZipFile(zip_path))
    with Image.open(io.BytesIO(archive.read(member))) as image:
        return np.array(image.convert("RGB"))


def _score_image_groups(
    arguments: tuple[
        list[tuple[str, pd.DataFrame]],
        dict[str, tuple[str, str]],
    ],
) -> list[dict[str, Any]]:
    image_groups, image_index = arguments
    archives: dict[str, zipfile.ZipFile] = {}
    rows: list[dict[str, Any]] = []
    try:
        for image_id, group in image_groups:
            image = load_source_image(
                image_id,
                group["source"].iloc[0],
                image_index,
                archives,
            )
            boxes = [json.loads(value) for value in group["bbox_xywh"]]
            for sample, score in zip(
                group.to_dict("records"),
                score_image(image, boxes),
                strict=True,
            ):
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "image_id": image_id,
                        "code": sample["code"],
                        "name": sample["name"],
                        "split": sample["split"],
                        "source": sample["source"],
                        **score,
                    }
                )
    finally:
        for archive in archives.values():
            archive.close()
    return rows


def _draw_review_sheet(
    items: list[tuple[int, str]],
    samples: pd.DataFrame,
    suspects: pd.DataFrame,
    priority: int,
    image_index: dict[str, tuple[str, str]],
    path: Path,
) -> None:
    archives: dict[str, zipfile.ZipFile] = {}
    tiles: list[np.ndarray] = []
    priority_by_sample = suspects.set_index("sample_id")["priority"]
    try:
        for number, image_id in items:
            group = samples[samples["image_id"] == image_id]
            image = load_source_image(
                image_id,
                group["source"].iloc[0],
                image_index,
                archives,
            )
            _, regions = find_pill_regions(image)
            rendered = image.copy()
            for x, y, width, height in regions:
                cv2.rectangle(
                    rendered,
                    (x, y),
                    (x + width, y + height),
                    (255, 0, 255),
                    3,
                )
            for sample in group.to_dict("records"):
                x, y, width, height = map(int, json.loads(sample["bbox_xywh"]))
                sample_priority = priority_by_sample.get(sample["sample_id"], 0)
                color = (
                    (255, 0, 0)
                    if sample_priority == priority
                    else (255, 140, 0)
                    if sample_priority
                    else (0, 200, 0)
                )
                cv2.rectangle(
                    rendered,
                    (x, y),
                    (x + width, y + height),
                    color,
                    7,
                )
                cv2.putText(
                    rendered,
                    sample["code"][-6:],
                    (x + 5, y + 45),
                    0,
                    1.6,
                    color,
                    4,
                )
            rendered = cv2.resize(rendered, (488, 640))
            cv2.putText(rendered, str(number), (8, 40), 0, 1.4, (255, 255, 0), 3)
            tiles.append(rendered)
    finally:
        for archive in archives.values():
            archive.close()

    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack(
        [np.hstack(tiles[index : index + 4]) for index in range(0, len(tiles), 4)]
    )
    if not cv2.imwrite(
        str(path),
        cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    ):
        raise OSError(f"failed to write review sheet: {path}")


def detect_bbox_errors(
    *,
    package_dir: Path = EXPORT / "training_package_classifier",
    records_dir: Path = LABEL_AUDIT_RECORDS,
    output_dir: Path = LABEL_AUDIT_OUTPUT,
    processes: int = 8,
    write_sheets: bool = True,
) -> pd.DataFrame:
    """전체 v1 crop의 위치 점수를 계산하고 사람 검수 후보를 갱신한다."""

    if processes <= 0:
        raise ValueError("processes must be greater than zero")
    samples = pd.read_csv(package_dir / "samples.csv")
    image_index = build_source_image_index()
    image_groups = list(samples.groupby("image_id"))
    chunks = [image_groups[index::processes] for index in range(processes)]
    arguments = [(chunk, image_index) for chunk in chunks if chunk]
    if processes == 1:
        rows = _score_image_groups(arguments[0])
    else:
        with Pool(processes) as pool:
            rows = [
                row for part in pool.map(_score_image_groups, arguments) for row in part
            ]
    scores = (
        pd.DataFrame(rows)
        .set_index("sample_id")
        .loc[samples["sample_id"]]
        .reset_index()
    )
    scores["priority"] = [position_priority(row) for row in scores.itertuples()]

    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    save_table(scores, output_dir / "position_scores.csv")
    suspects = scores[scores["priority"] > 0].sort_values(
        ["priority", "match_iou", "fg_in"]
    )
    image_numbers: dict[tuple[int, str], int] = {}
    for priority in (1, 2, 3):
        image_ids = dict.fromkeys(
            suspects.loc[suspects["priority"] == priority, "image_id"]
        )
        for number, image_id in enumerate(image_ids):
            image_numbers[(priority, image_id)] = number
    suspects = suspects.copy()
    suspects.insert(
        0,
        "그림번호",
        [
            image_numbers[(priority, image_id)]
            for priority, image_id in zip(
                suspects["priority"],
                suspects["image_id"],
                strict=True,
            )
        ],
    )
    suspects.insert(
        1,
        "그림파일",
        [
            f"review/p{priority}_review_{number // 12 + 1}.jpg"
            for priority, number in zip(
                suspects["priority"],
                suspects["그림번호"],
                strict=True,
            )
        ],
    )
    review_path = records_dir / "position_suspects.csv"
    suspects = preserve_human_reviews(suspects, review_path)
    save_table(suspects, review_path)

    if write_sheets:
        review_dir = output_dir / "review"
        review_dir.mkdir(exist_ok=True)
        for old_path in review_dir.glob("p*_review_*.jpg"):
            old_path.unlink()
        for priority in (1, 2, 3):
            image_ids = list(
                dict.fromkeys(
                    suspects.loc[suspects["priority"] == priority, "image_id"]
                )
            )
            for start in range(0, len(image_ids), 12):
                _draw_review_sheet(
                    list(enumerate(image_ids[start : start + 12], start)),
                    samples,
                    suspects,
                    priority,
                    image_index,
                    review_dir / f"p{priority}_review_{start // 12 + 1}.jpg",
                )
    return suspects


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-dir", type=Path, default=EXPORT / "training_package_classifier"
    )
    parser.add_argument("--records-dir", type=Path, default=LABEL_AUDIT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=LABEL_AUDIT_OUTPUT)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args(argv)
    suspects = detect_bbox_errors(
        package_dir=args.package_dir,
        records_dir=args.records_dir,
        output_dir=args.output_dir,
        processes=args.processes,
        write_sheets=not args.no_sheets,
    )
    print(f"위치 오류 후보 bbox {len(suspects)}개")


if __name__ == "__main__":
    main()
