"""Kaggle + AI Hub 조합 이미지 중에서 클래스별 bbox가 목표 개수(기본 108)가 되도록 이미지를 고른다.

규칙
- 촬영 조건: 연회색 배경, 주백색 조명, 70/75/90° (test와 같은 조건)
- 제외: validate_labels.py가 정한 라벨 오류 (조합은 이미지째, Kaggle은 해당 bbox만). 먼저 실행해야 한다
- Kaggle은 전부 쓰고, 모자란 만큼만 조합 이미지로 채운다. 목표에 못 미치는 클래스는 있는 만큼만 쓴다.
- 고르는 방법(정수계획, scipy milp): 목표를 넘는 bbox 수를 최소화.
  각 각도가 30% 이상이 되게 하고, 같은 조합보다 서로 다른 조합을 우선한다.

결과
    data/classifier/meta/selected_images.csv  image, source(kaggle/combo), zip, group(조합), angle, classes
    data/classifier/meta/boxes.csv            image, code, x, y, w, h, W, H (픽셀)
    data/classifier/meta/class_summary.csv    클래스별 kaggle, combo, total, 조합 가용 수

사용:
    python -m scripts.classifier_data.select_images [--target 108]
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from scripts.classifier_data.paths import (
    CLASSES,
    COMBO_ZIPS,
    KAGGLE,
    META,
    save_table,
    tl_zip,
)

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=int, default=108)
args = ap.parse_args()
T = args.target

classes = pd.read_csv(CLASSES)
codes = list(classes.code)


iss = pd.read_csv(META / "label_issues.csv")
ex_img = set(iss[iss.action == "이미지 제외"].image)
ex_box = set(
    zip(
        iss[iss.action == "bbox 제외"].image,
        iss[iss.action == "bbox 제외"].code,
        strict=True,
    )
)

# 조합: 약마다 JSON이 따로 있다 ({조합}_json/{약 코드}/{이미지}.json)
labels = defaultdict(list)  # image -> [(code, bbox, W, H)]
zip_of, n_all = {}, set()
for n in COMBO_ZIPS:
    z = tl_zip(n)
    for m in z.namelist():
        if not m.endswith(".json") or "-오류" in m or "/._" in m:
            continue
        q = os.path.basename(m)[:-5].split("_")
        if len(q) != 8 or not (
            q[2] == "2" and q[4] == "2" and q[5] in ("70", "75", "90")
        ):
            continue
        image = os.path.basename(m)[:-5] + ".png"
        n_all.add(image)
        if image in ex_img:
            continue
        j = json.loads(z.read(m))
        W, H = j["images"][0]["width"], j["images"][0]["height"]
        for a in j["annotations"]:
            labels[image].append((m.split("/")[-2], a.get("bbox"), W, H))
        zip_of[image] = f"TS_{n}_조합"
print(f"조합 조건 부합 {len(n_all)}장, 검수에서 제외 {len(n_all) - len(labels)}장")
labels = {i: v for i, v in labels.items() if all(c in codes for c, *_ in v)}

# Kaggle: 전부 사용하되 검수에서 뺀 bbox만 거른다
kaggle = defaultdict(list)
for f in KAGGLE.glob("train_annotations/*/*/*.json"):
    if f.stem + ".png" in ex_img:
        continue
    with f.open(encoding="utf-8") as file:
        j = json.load(file)
    W, H = j["images"][0]["width"], j["images"][0]["height"]
    for a in j["annotations"]:
        if (f.stem + ".png", f.parent.name) not in ex_box:
            kaggle[f.stem + ".png"].append((f.parent.name, a["bbox"], W, H))
K = Counter(c for v in kaggle.values() for c, *_ in v)
assert set(K) <= set(codes), f"classes.csv에 없는 Kaggle 클래스: {set(K) - set(codes)}"
avail = Counter(c for v in labels.values() for c, *_ in v)
print(
    f"Kaggle {len(kaggle)}장 bbox {sum(K.values())}개, 조합 후보 {len(labels)}장 bbox {sum(avail.values())}개"
)

# 정수계획: x(이미지 선택), o(클래스별 목표 초과), e(선택된 조합 수)
candidate_images = sorted(labels)
groups = defaultdict(list)
for i, b in enumerate(candidate_images):
    groups[b.split("_")[0]].append(i)
G = list(groups)
nI, nC, nG = len(candidate_images), len(codes), len(G)
cidx = {c: j for j, c in enumerate(codes)}
iO, iE = nI, nI + nC
nV = iE + nG
A = lil_matrix((2 * nC + nG + 3, nV))
lo, hi = [], []
for i, b in enumerate(candidate_images):
    for c, *_ in labels[b]:
        A[cidx[c], i] += 1
        A[nC + cidx[c], i] += 1
for c in codes:  # 조합 bbox >= 목표까지 모자란 수 (가용 범위 안에서)
    lo.append(min(T, K[c] + avail[c]) - K[c])
    hi.append(np.inf)
for c, j in cidx.items():  # 조합 bbox - 초과 <= 목표 - Kaggle
    A[nC + j, iO + j] = -1
    lo.append(-np.inf)
    hi.append(T - K[c])
for k, g in enumerate(G):  # 조합 g에서 하나라도 고르면 e_g = 1
    for i in groups[g]:
        A[2 * nC + k, i] = 1
    A[2 * nC + k, iE + k] = -len(groups[g])
    lo.append(-np.inf)
    hi.append(0)
r0 = 2 * nC + nG
for t, ang in enumerate(["70", "75", "90"]):  # 각 각도 30% 이상
    for i, b in enumerate(candidate_images):
        A[r0 + t, i] = (b.split("_")[5] == ang) - 0.30
    lo.append(0)
    hi.append(np.inf)
cost = np.zeros(nV)
cost[:nI] = 0.01  # 이미지는 적게
cost[iO:iE] = 1  # 목표 초과는 가장 크게 벌점
cost[iE:] = -0.02  # 서로 다른 조합은 이득
ub = np.full(nV, np.inf)
ub[:nI] = 1
ub[iE:] = 1
integ = np.zeros(nV)
integ[:nI] = 1
integ[iE:] = 1
res = milp(
    cost,
    constraints=LinearConstraint(A.tocsr(), lo, hi),
    integrality=integ,
    bounds=Bounds(np.zeros(nV), ub),
    options={"time_limit": 300},
)
assert res.x is not None, res.message
print("최적화:", res.message)
sel = [b for i, b in enumerate(candidate_images) if res.x[i] > 0.5]

# 저장
META.mkdir(parents=True, exist_ok=True)
rows, boxes = [], []
for image, v, src in [(i, kaggle[i], "kaggle") for i in sorted(kaggle)] + [
    (i, labels[i], "combo") for i in sel
]:
    rows.append(
        {
            "image": image,
            "source": src,
            "zip": zip_of.get(image, "kaggle"),
            "group": image.split("_")[0],
            "angle": image.split("_")[5],
            "classes": "|".join(sorted(c for c, *_ in v)),
        }
    )
    boxes += [
        {
            "image": image,
            "code": c,
            "x": b[0],
            "y": b[1],
            "w": b[2],
            "h": b[3],
            "W": W,
            "H": H,
        }
        for c, b, W, H in v
    ]
save_table(pd.DataFrame(rows), META / "selected_images.csv")
save_table(pd.DataFrame(boxes), META / "boxes.csv")

cc = Counter(c for b in sel for c, *_ in labels[b])
sm = classes[["code", "name"]].copy()
sm["kaggle"] = sm.code.map(K).fillna(0).astype(int)
sm["combo"] = sm.code.map(cc).fillna(0).astype(int)
sm["total"] = sm.kaggle + sm.combo
sm["combo_available"] = sm.code.map(avail).fillna(0).astype(int)
save_table(sm.sort_values("total"), META / "class_summary.csv")

print(
    f"선택: 조합 이미지 {len(sel)}장(조합 {len({b.split('_')[0] for b in sel})}개), Kaggle {len(kaggle)}장"
)
print("각도:", dict(Counter(b.split("_")[5] for b in sel)))
print(
    f"bbox {sm.total.sum()}개, 클래스별 최소 {sm.total.min()}, 중앙값 {sm.total.median():.0f}, 최대 {sm.total.max()}"
)
print(f"목표 {T} 미만 {(sm.total < T).sum()}종:")
print(
    sm[sm.total < T][["code", "name", "kaggle", "combo", "total"]].to_string(
        index=False
    )
)
