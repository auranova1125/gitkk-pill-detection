"""라벨 2차 검수 ①: 사진에서 찾은 알약 영역과 bbox를 짝 맞춰 위치가 틀린 bbox를 찾는다.

1차 검수(validate_labels.py)는 bbox 값만 보므로, 알약이 없는 곳이나 다른 알약 위에 그려진 bbox를 잡지 못한다.
여기서는 사진 픽셀을 본다.

방법 (사진마다, 1/4 해상도)
    1. 배경: 조명 그라데이션이 있으므로 Lab 색을 2차 곡면으로 맞춘다 (알약 픽셀은 반복해서 뺀다)
    2. 알약 영역: 배경과 색이 다른 픽셀 + 윤곽선(배경과 색이 비슷한 파란·회색 알약용) → 덩어리 안을 채운다
       잡티와 사진 가장자리의 비네팅 덩어리는 뺀다
    3. bbox와 덩어리를 IoU로 1:1 짝 맞춘다 (헝가리안)

점수 (bbox마다)
    match_iou        짝이 된 덩어리와의 IoU. 낮을수록 의심
    fg_in            bbox 안에서 알약 영역 비율. 0에 가까우면 배경에 친 bbox
    ring_fg          bbox를 8% 넓힌 바깥 띠의 알약 영역 비율 (다른 bbox 영역 제외). 높으면 알약이 잘렸을 수 있음
    unmatched_blobs  같은 사진에서 bbox와 짝이 안 된 덩어리 수. bbox가 없는 알약일 수 있음

의심 등급
    1  강함   배경 bbox(fg_in < 0.15), 밀린 bbox(0 < match_iou < 0.5),
              짝 없음 + bbox 없는 알약 있음, 짝 없음 + bbox 안이 반 이상 배경
    2  약함   짝 없음이지만 bbox 안이 알약으로 차 있음. 붙어 있는 알약이 한 덩어리로 잡힌 경우가 많다
    3  경계   0.5 ≤ match_iou < 0.6

결과
    data/classifier/meta/label_audit/position_scores.csv    bbox마다 점수 (전체)
    data/classifier/meta/label_audit/position_suspects.csv  의심 bbox와 검수 칸(판정, 조치)
    data/classifier/meta/label_audit/sheets/p<등급>_<번호>.jpg  검수용 그림
        초록·빨강 = 라벨 bbox(빨강이 의심, 숫자는 match_iou), 분홍 = 찾은 알약 영역

사용:
    python classifier_data_build/audit_bbox_position.py [--no-sheets]
"""
import argparse
import io
import json
import os
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

from config import COMBO_ZIPS, EXPORT, KAGGLE, META, save_table, ts_zip

S = 4                 # 축소 배율
FG = 12               # 배경과 다른 정도 기준
MIN_AREA = 0.004      # 이보다 작은 덩어리는 잡티 (사진 면적 대비)
EDGE_AREA = 0.02      # 가장자리에 닿고 이보다 작으면 비네팅
RING = 0.08           # 바깥 띠 폭 (bbox 긴 변 대비)
OUT = META / "label_audit"


def background_diff(im):
    """배경(2차 곡면)과 다른 정도. 그림자처럼 어두워지기만 하는 픽셀은 약하게 본다."""
    sm = cv2.resize(im, (im.shape[1] // S, im.shape[0] // S), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    X, Y = (xx / W - .5).ravel(), (yy / H - .5).ravel()
    A = np.stack([np.ones_like(X), X, Y, X * X, Y * Y, X * Y], 1)
    v = lab.reshape(-1, 3)
    keep = np.ones(len(v), bool)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
        res = v - A @ coef
        dist = np.linalg.norm(res, axis=1)
        keep = dist < max(3 * np.median(dist[keep]), 4)
    res = res.reshape(H, W, 3)
    dL, chroma = res[..., 0], np.hypot(res[..., 1], res[..., 2])
    return np.maximum.reduce([chroma * 1.5, np.maximum(dL, 0), np.maximum(-dL - 25, 0)]), sm


def pill_regions(im):
    """(채운 알약 영역 마스크, 덩어리 bbox 목록[원본 좌표])"""
    diff, sm = background_diff(im)
    fg = (diff > FG).astype(np.uint8)
    edge = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY), (3, 3), 0), 20, 50)
    fg |= cv2.dilate(edge, np.ones((3, 3), np.uint8)) > 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = fg.shape
    filled, regions = np.zeros_like(fg), []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        on_edge = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        if a < MIN_AREA * H * W or (on_edge and a < EDGE_AREA * H * W):
            continue
        cv2.drawContours(filled, [c], -1, 1, -1)
        regions.append([x * S, y * S, w * S, h * S])
    return filled, regions


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i)


def score_image(im, boxes):
    """bbox마다 점수 dict 목록"""
    filled, regions = pill_regions(im)
    M = np.array([[iou(b, r) for r in regions] for b in boxes]) if regions else np.zeros((len(boxes), 0))
    match = {}
    if M.size:
        for i, j in zip(*linear_sum_assignment(-M)):
            if M[i, j] > 0:
                match[i] = M[i, j]
    unmatched = len(regions) - len(match)
    out = []
    for i, b in enumerate(boxes):
        x, y, w, h = [int(v / S) for v in b]
        x2, y2 = x + max(w, 1), y + max(h, 1)
        m = max(1, int(RING * max(w, h)))
        ring = np.zeros(filled.shape, bool)
        ring[max(0, y - m):y2 + m, max(0, x - m):x2 + m] = True
        ring[y:y2, x:x2] = False
        for j, o in enumerate(boxes):
            if j != i:
                ox, oy, ow, oh = [int(v / S) for v in o]
                ring[oy:oy + oh, ox:ox + ow] = False
        out.append(dict(
            fg_in=round(float(filled[y:y2, x:x2].mean()), 3),
            ring_fg=round(float(filled[ring].mean()) if ring.any() else 0.0, 3),
            match_iou=round(float(match.get(i, 0.0)), 3),
            n_boxes=len(boxes), n_regions=len(regions), unmatched_blobs=unmatched))
    return out


def priority(r):
    if r.fg_in < 0.15 or 0 < r.match_iou < 0.5 or (r.match_iou == 0 and (r.unmatched_blobs > 0 or r.fg_in < 0.5)):
        return 1
    if r.match_iou == 0:
        return 2
    if r.match_iou < 0.6:
        return 3
    return 0


# 원본 사진 위치: 파일명 -> (zip 경로, zip 안 경로). make_crops.py와 같은 방식
def build_index():
    where = {}
    for n in COMBO_ZIPS:
        p = ts_zip(n)
        with zipfile.ZipFile(p) as z:
            for m in z.namelist():
                if m.endswith(".png"):
                    where[os.path.basename(m)] = (str(p), m)
    return where


def load_image(image_id, source, where, zips):
    name = image_id + ".png"
    if source == "kaggle":
        return np.array(Image.open(KAGGLE / "train_images" / name).convert("RGB"))
    p, m = where[name]
    z = zips.setdefault(p, zipfile.ZipFile(p))
    return np.array(Image.open(io.BytesIO(z.read(m))).convert("RGB"))


def run_group(args):
    images, where = args
    zips, rows = {}, []
    for image_id, g in images:
        im = load_image(image_id, g.source.iloc[0], where, zips)
        boxes = [json.loads(b) for b in g.bbox_xywh]
        for r, sc in zip(g.itertuples(), score_image(im, boxes)):
            rows.append(dict(sample_id=r.sample_id, image_id=image_id, code=r.code, name=r.name,
                             split=r.split, source=r.source, **sc))
    return rows


def draw_sheet(items, samples, scores, where, path):
    """의심 사진 24장을 6열로 그린다"""
    zips, tiles = {}, []
    for k, image_id in items:
        g = samples[samples.image_id == image_id]
        im = load_image(image_id, g.source.iloc[0], where, zips)
        _, regions = pill_regions(im)
        o = im.copy()
        for x, y, w, h in regions:
            cv2.rectangle(o, (x, y), (x + w, y + h), (255, 0, 255), 3)
        for r in g.itertuples():
            x, y, w, h = map(int, json.loads(r.bbox_xywh))
            sc = scores.loc[r.sample_id]
            col = (255, 0, 0) if sc.priority else (0, 200, 0)
            cv2.rectangle(o, (x, y), (x + w, y + h), col, 6)
            cv2.putText(o, f"{r.code[-6:]} {sc.match_iou:.2f}", (x, max(30, y - 8)), 0, 1.3, col, 3)
        o = cv2.resize(o, (366, 480))
        cv2.putText(o, f"{k} {image_id[:29]}", (4, 472), 0, 0.45, (255, 255, 0), 1)
        tiles.append(o)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sheets", action="store_true", help="검수용 그림을 만들지 않는다")
    args = ap.parse_args()

    samples = pd.read_csv(EXPORT / "training_package_classifier" / "samples.csv")
    where = build_index()
    images = list(samples.groupby("image_id"))
    chunks = [(images[i::8], where) for i in range(8)]
    with Pool(8) as pool:
        rows = [r for part in pool.map(run_group, chunks) for r in part]
    scores = pd.DataFrame(rows).set_index("sample_id").loc[samples.sample_id].reset_index()
    scores["priority"] = [priority(r) for r in scores.itertuples()]

    OUT.mkdir(parents=True, exist_ok=True)
    save_table(scores, OUT / "position_scores.csv")
    sus = scores[scores.priority > 0].sort_values(["priority", "match_iou", "fg_in"]).copy()
    sus["판정"], sus["조치"], sus["메모"] = "", "", ""
    save_table(sus, OUT / "position_suspects.csv")

    print(f"bbox {len(scores)}개 / 사진 {scores.image_id.nunique()}장")
    for p in (1, 2, 3):
        s = sus[sus.priority == p]
        print(f"  {p}등급: bbox {len(s)}개, 사진 {s.image_id.nunique()}장 (train {(s.split == 'train').sum()}, val {(s.split == 'val').sum()})")

    if not args.no_sheets:
        (OUT / "sheets").mkdir(exist_ok=True)
        for old in (OUT / "sheets").glob("*.jpg"):
            old.unlink()
        sc = scores.set_index("sample_id")
        for p in (1, 2, 3):
            ids = list(dict.fromkeys(sus[sus.priority == p].image_id))
            for n in range(0, len(ids), 24):
                draw_sheet(list(enumerate(ids[n:n + 24], n)), samples, sc, where, OUT / "sheets" / f"p{p}_{n // 24 + 1}.jpg")
        print(f"검수 그림: {OUT / 'sheets'}")
