"""「알약 데이터 검증 체크리스트」대로 라벨을 검사하고, 이미지 선정 전에 뺄 이미지와 bbox를 정한다.

대상: select_source_images.py의 후보 전체
    - 조합 TL_1·3~8 중 촬영 조건(연회색 배경, 주백색 조명, 70/75/90°)에 맞는 이미지
    - Kaggle train 전부

처리 (action)
    이미지 제외  조합 이미지에서 라벨 오류가 하나라도 있으면 이미지째 뺀다 (대신 고를 조합 이미지가 많다)
    bbox 제외    Kaggle은 대신할 이미지가 없으므로 문제 있는 bbox만 뺀다
    기록만       crop 라벨에는 영향이 없거나, 오류가 아니라 분포 특성인 것

결과
    data/classifier/meta/label_issues.csv      문제 하나당 한 줄 (source, image, code, check, level, action, detail)
    reports/data/classifier/validation_report.md  항목별 건수와 처리 (기록용)

사용:
    python -m etl.classifier.validate_source_labels [--skip-png]
"""

import argparse
import hashlib
import io
import itertools
import json
import os
import zipfile
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from etl.classifier.label_checklist import CHECKS
from etl.classifier.paths import (
    COMBO_ZIPS,
    KAGGLE,
    META,
    REPORTS,
    save_table,
    tl_zip,
    ts_zip,
)


def validate_source_labels(*, skip_png: bool = False) -> None:
    RATIO = (18, 33)  # 1mm당 픽셀 정상 범위 (체크리스트 예시값, Train 중앙값 25.3)
    IOU = 0.7
    CONTAIN = 0.9  # 작은 bbox 면적 중 다른 bbox와 겹친 비율. 이 이상이면 한 알약에 라벨 두 개로 본다
    PHASH = 5
    REL = (
        0.7,
        0.5,
    )  # 실제 크기 값(leng_long)이 없는 약: 클래스 bbox 중앙값 대비 긴 변·짧은 변 최소 배율
    # (크기 검사가 되는 116종의 선정 bbox는 최소 0.76·0.58배)

    LEVEL = {c: lv for c, _, lv in CHECKS}

    # ── 라벨 읽기: bbox 하나당 한 줄 ──────────────────────────────────────────
    recs, issues = [], []
    files = []  # 검수 대상 라벨 JSON 목록 (약 하나당 한 파일). export_training_package.py가 item_seq·drug_dir를 읽을 때 쓴다

    def issue(source, image, code, check, action, detail=""):
        issues.append(
            {
                "source": source,
                "image": image,
                "code": code,
                "check": check,
                "level": LEVEL[check],
                "action": action,
                "detail": detail,
            }
        )

    def add(j, source, image, code):
        im = j["images"][0]
        for a in j["annotations"] or [
            {}
        ]:  # annotation이 비어 있어도 한 줄 남겨 검사에 걸리게 한다
            recs.append(
                {
                    "source": source,
                    "image": image,
                    "code": code,
                    "W": im.get("width"),
                    "H": im.get("height"),
                    "bbox": a.get("bbox"),
                    "area": a.get("area"),
                    "iscrowd": a.get("iscrowd", 0),
                    "ignore": a.get("ignore", 0),
                    "ann_id": a.get("id"),
                    "img_id": im.get("id"),
                    "cat_id": a.get("category_id"),
                    "drug_N": im.get("drug_N"),
                    "dl_idx": im.get("dl_idx"),
                    "map_code": im.get("dl_mapping_code"),
                    "dl_name": im.get("dl_name"),
                    "shape": im.get("drug_shape"),
                    "color": im.get("color_class1"),
                    "print_front": im.get("print_front"),
                    "leng": im.get("leng_long"),
                    "leng_s": im.get("leng_short"),
                }
            )

    for n in COMBO_ZIPS:
        z = tl_zip(n)
        for m in z.namelist():
            if not m.endswith(".json") or "/._" in m:
                continue
            stem = os.path.basename(m)[:-5]
            q = stem.split("_")
            if len(q) != 8 or not (
                q[2] == "2" and q[4] == "2" and q[5] in ("70", "75", "90")
            ):
                continue
            image, code = stem + ".png", m.split("/")[-2]
            files.append(
                {
                    "source": "combo",
                    "image": image,
                    "code": code,
                    "json": f"{os.path.basename(z.filename)}/{m}",
                }
            )
            if "-오류" in m:  # AI Hub가 오류로 표시한 폴더
                issue(
                    "combo",
                    image,
                    code,
                    "파일명 ↔ 라벨 구성 불일치",
                    "이미지 제외",
                    "AI Hub -오류 폴더",
                )
                continue
            try:
                add(json.loads(z.read(m)), "combo", image, code)
            # 손상된 원본 annotation 하나가 전체 검수를 중단하지 않게 기록한다.
            except Exception as e:
                issue(
                    "combo",
                    image,
                    code,
                    "bbox 음수·0 크기",
                    "이미지 제외",
                    f"JSON 읽기 실패: {e}",
                )

    for f in sorted(KAGGLE.glob("train_annotations/*/*/*.json")):
        files.append(
            {
                "source": "kaggle",
                "image": f.stem + ".png",
                "code": f.parent.name,
                "json": str(f.relative_to(KAGGLE.parent)),
            }
        )
        with f.open(encoding="utf-8") as file:
            add(json.load(file), "kaggle", f.stem + ".png", f.parent.name)

    d = pd.DataFrame(recs)
    DROP = {"combo": "이미지 제외", "kaggle": "bbox 제외"}
    print(
        f"라벨 읽음: 조합 {d[d.source == 'combo'].image.nunique()}장, Kaggle {d[d.source == 'kaggle'].image.nunique()}장, bbox {len(d)}개"
    )

    def valid(b):
        return (
            isinstance(b, list)
            and len(b) == 4
            and all(isinstance(v, (int, float)) for v in b)
            and b[2] > 0
            and b[3] > 0
        )

    def ratio(side, leng):
        """1mm당 픽셀. 길이(mm) 값이 없으면 None."""
        try:
            L = float(leng)
        except (TypeError, ValueError):
            return None
        return side / L if L > 0 else None

    def overlap(a, b):
        """(IoU, 작은 bbox 면적 중 겹친 비율)"""
        ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        i = ix * iy
        return i / (a[2] * a[3] + b[2] * b[3] - i), i / min(a[2] * a[3], b[2] * b[3])

    d["ok"] = d.bbox.map(valid)
    d["ratio"] = pd.to_numeric(
        [ratio(max(r.bbox[2:]), r.leng) if r.ok else None for r in d.itertuples()]
    )
    # 짧은 변: 알약이 기울면 bbox 짧은 변은 알약 폭보다 길어지지만 짧아지지는 않는다 → 아래쪽 기준만 본다
    d["ratio_s"] = pd.to_numeric(
        [ratio(min(r.bbox[2:]), r.leng_s) if r.ok else None for r in d.itertuples()]
    )
    d["ratio_ok"] = d.ratio.between(*RATIO) & ~(d.ratio_s < RATIO[0])
    d["long"] = [max(b[2:]) if ok else None for b, ok in zip(d.bbox, d.ok, strict=True)]
    d["short"] = [
        min(b[2:]) if ok else None for b, ok in zip(d.bbox, d.ok, strict=True)
    ]
    med = (
        d[d.ok].groupby("code")[["long", "short"]].median()
    )  # leng_long이 없을 때 비교 기준

    # ── ① Annotation ─────────────────────────────────────────────────────────
    for r in d.itertuples():
        if not r.ok:
            issue(
                r.source,
                r.image,
                r.code,
                "bbox 음수·0 크기",
                DROP[r.source],
                f"bbox={r.bbox}",
            )
            continue
        x, y, w, h = r.bbox
        if not isinstance(r.area, (int, float)) or abs(r.area - w * h) > 1:
            issue(
                r.source,
                r.image,
                r.code,
                "bbox 면적 불일치",
                "기록만",
                f"area={r.area}, w*h={w * h} (crop은 bbox만 씀)",
            )
        if x < 0 or y < 0 or x + w > r.W or y + h > r.H:
            inside = (
                max(0, min(r.W, x + w) - max(0, x))
                * max(0, min(r.H, y + h) - max(0, y))
                / (w * h)
            )
            act = DROP[r.source] if inside < 0.7 else "기록만"
            issue(
                r.source,
                r.image,
                r.code,
                "bbox 범위 초과",
                act,
                f"이미지 안 {inside:.0%}"
                + ("" if inside < 0.7 else " (crop 때 경계에 맞춤)"),
            )
        if r.iscrowd or r.ignore:
            issue(
                r.source,
                r.image,
                r.code,
                "ignore·iscrowd 이상",
                "기록만",
                f"iscrowd={r.iscrowd}, ignore={r.ignore}",
            )

    # Annotation·이미지 ID: 조합은 약마다 JSON이 따로라 id가 항상 1 → Kaggle만 검사
    k = d[d.source == "kaggle"]
    for aid, g in k.groupby("ann_id"):
        if (
            g.apply(
                lambda r: (r.img_id, tuple(r.bbox or []), r.cat_id), axis=1
            ).nunique()
            > 1
        ):
            for r in g.itertuples():
                issue(
                    "kaggle",
                    r.image,
                    r.code,
                    "Annotation ID 중복",
                    "기록만",
                    f"id={aid}",
                )
    for col, other in (("img_id", "image"), ("image", "img_id")):
        for key, g in k.groupby(col):
            if g[other].nunique() > 1:
                for img in g.image.unique():
                    issue(
                        "kaggle",
                        img,
                        "",
                        "이미지 ID 중복",
                        "기록만",
                        f"{col}={key} → {other} {g[other].nunique()}개",
                    )

    # 같은 이미지 안의 동일 bbox / 과도한 중첩
    for img, g in d[d.ok].groupby("image"):
        for a, b in itertools.combinations(g.itertuples(), 2):
            same = a.bbox == b.bbox
            v, c = overlap(a.bbox, b.bbox)
            if not same and v <= IOU and c < CONTAIN:
                continue
            check = "동일 bbox 반복" if same else "bbox 과도한 중첩"
            for p, q in ((a, b), (b, a)):
                act = DROP[p.source]
                # Kaggle: 둘 중 하나만 크기가 맞으면 그쪽 라벨이 맞다고 보고 남긴다
                if p.source == "kaggle" and p.ratio_ok and not q.ratio_ok:
                    act = "기록만"
                issue(
                    p.source,
                    img,
                    p.code,
                    check,
                    act,
                    f"{q.code}와 IoU {v:.2f}, 작은 bbox의 {c:.0%} 겹침",
                )

    # ── ③ 교차 검증 ──────────────────────────────────────────────────────────
    for r in d.itertuples():
        if r.source == "combo":
            # 조합 JSON의 category_id는 항상 1이고 dl_idx도 코드와 규칙이 달라(대부분 코드-1) 쓸 수 없다
            ok = r.code == r.drug_N == r.map_code
        else:
            ok = (
                r.cat_id is not None
                and f"K-{int(r.cat_id):06d}" == r.code == r.drug_N
                and str(r.dl_idx) == str(r.cat_id)
            )
        if not ok:
            issue(
                r.source,
                r.image,
                r.code,
                "코드 매핑 불일치",
                DROP[r.source],
                f"category_id={r.cat_id}, dl_idx={r.dl_idx}, drug_N={r.drug_N}, dl_mapping_code={r.map_code}",
            )

    for img, g in d.groupby("image"):
        src = g.source.iloc[0]
        fc = {f"K-{c}" for c in img.split("_")[0].split("-")[1:]}
        ac = set(g.code)
        for c in sorted(fc - ac):
            # 라벨 없는 알약은 crop 라벨을 틀리게 하진 않지만, 조합은 라벨 품질이 의심되므로 대체 가능한 이미지로 바꾼다
            issue(
                src,
                img,
                c,
                "파일명 ↔ 라벨 구성 불일치",
                "이미지 제외" if src == "combo" else "기록만",
                "파일명에 있는데 라벨 없음",
            )
        for c in sorted(ac - fc):
            issue(
                src,
                img,
                c,
                "파일명 ↔ 라벨 구성 불일치",
                DROP[src],
                "라벨에 있는데 파일명에 없음",
            )

    for code, g in d.groupby("code"):
        diff = {
            c: sorted(g[c].dropna().astype(str).unique())
            for c in ("dl_name", "shape", "color", "print_front")
        }
        diff = {c: v for c, v in diff.items() if len(v) > 1}
        if diff:
            issue("all", "", code, "같은 클래스 제품 정보 상충", "기록만", str(diff))

    d["group"] = d.image.str.split("_").str[0]
    for (src, _group), g in d.groupby(["source", "group"]):
        n = g.groupby("image").size()
        if n.nunique() > 1:
            for img in n[n < n.max()].index:
                issue(
                    src,
                    img,
                    "",
                    "같은 조합의 각도별 알약 수 다름",
                    "기록만",
                    f"{n[img]}개 (같은 조합 최대 {n.max()}개)",
                )

    for r in d[d.ok].itertuples():
        if pd.isna(r.ratio):
            # 실제 크기와 비교할 수 없으면 같은 클래스 bbox 중앙값과 비교해 알약 일부만 감싼 bbox를 찾는다
            rl, rs = r.long / med.long[r.code], r.short / med.short[r.code]
            if rl < REL[0] or rs < REL[1]:
                issue(
                    r.source,
                    r.image,
                    r.code,
                    "bbox 크기 ↔ 실제 알약 크기 불일치",
                    DROP[r.source],
                    f"leng_long 값 없음 → 클래스 중앙값 대비 긴 변 {rl:.2f}배·짧은 변 {rs:.2f}배 (기준 {REL[0]}·{REL[1]}배 이상)",
                )
            else:
                issue(
                    r.source,
                    r.image,
                    r.code,
                    "bbox 크기 ↔ 실제 알약 크기 불일치",
                    "기록만",
                    f"leng_long 값 없음 ({r.leng}), 클래스 중앙값 대비 정상 (긴 변 {rl:.2f}배·짧은 변 {rs:.2f}배)",
                )
        elif not RATIO[0] <= r.ratio <= RATIO[1]:
            issue(
                r.source,
                r.image,
                r.code,
                "bbox 크기 ↔ 실제 알약 크기 불일치",
                DROP[r.source],
                f"긴 변 {r.ratio:.1f}px/mm (정상 {RATIO[0]}~{RATIO[1]}, leng_long {r.leng}mm)",
            )
        elif r.ratio_s < RATIO[0]:
            issue(
                r.source,
                r.image,
                r.code,
                "bbox 크기 ↔ 실제 알약 크기 불일치",
                DROP[r.source],
                f"짧은 변 {r.ratio_s:.1f}px/mm (정상 {RATIO[0]} 이상, leng_short {r.leng_s}mm)",
            )

    # ── EDA (기록만) ──────────────────────────────────────────────────────────
    e = d[d.ok].copy()
    e["area_px"] = e.bbox.map(lambda b: b[2] * b[3])
    e["aspect"] = e.bbox.map(lambda b: b[2] / b[3])
    for code, g in e.groupby("code"):
        for col, name in (("area_px", "넓이"), ("aspect", "종횡비")):
            q1, q3 = g[col].quantile([0.25, 0.75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            for r in g[(g[col] < lo) | (g[col] > hi)].itertuples():
                issue(
                    r.source,
                    r.image,
                    code,
                    "클래스별 bbox 크기·종횡비 이상치",
                    "기록만",
                    f"{name} {getattr(r, col):.2f} (정상 {lo:.2f}~{hi:.2f})",
                )
        cell = g.apply(
            lambda r: (
                int((r.bbox[0] + r.bbox[2] / 2) / r.W * 2),
                int((r.bbox[1] + r.bbox[3] / 2) / r.H * 2),
            ),
            axis=1,
        )
        top = cell.value_counts(normalize=True)
        if len(g) >= 10 and top.iloc[0] >= 0.7:
            issue(
                "all",
                "",
                code,
                "클래스별 bbox 위치 편향",
                "기록만",
                f"{top.index[0]} 칸에 {top.iloc[0]:.0%} (crop하면 위치 정보는 사라짐)",
            )

    # ── ② Image: PNG를 열어 확인 ─────────────────────────────────────────────
    if not skip_png:
        where = {}
        for n in COMBO_ZIPS:
            z = zipfile.ZipFile(ts_zip(n))
            for m in z.namelist():
                if m.endswith(".png") and "/._" not in m:
                    where[os.path.basename(m)] = (z, m)
        size = d.groupby("image")[["W", "H", "source"]].first()
        md5, ph = {}, {}
        for i, (img, r) in enumerate(size.iterrows(), 1):
            try:
                if r.source == "kaggle":
                    raw = (KAGGLE / "train_images" / img).read_bytes()
                else:
                    z, m = where[img]
                    raw = z.read(m)
            except (KeyError, FileNotFoundError):
                issue(r.source, img, "", "PNG 누락·손상", "이미지 제외", "PNG 없음")
                continue
            try:
                im = Image.open(io.BytesIO(raw))
                im.load()
            except (OSError, ValueError) as ex:
                issue(
                    r.source,
                    img,
                    "",
                    "PNG 누락·손상",
                    "이미지 제외",
                    f"열 수 없음: {ex}",
                )
                continue
            if im.size != (r.W, r.H):
                issue(
                    r.source,
                    img,
                    "",
                    "PNG 크기 불일치",
                    "이미지 제외",
                    f"PNG {im.size}, JSON {(r.W, r.H)}",
                )
            md5[img] = hashlib.md5(im.tobytes()).hexdigest()
            c = dctn(
                np.asarray(
                    im.convert("L").resize((32, 32), Image.LANCZOS), dtype=float
                ),
                norm="ortho",
            )[:8, :8]
            ph[img] = np.packbits(c.flatten() > np.median(c)).view(">u8")[0]
            if i % 1000 == 0:
                print(f"  PNG {i}/{len(size)}")

        by = defaultdict(list)
        for img, h in md5.items():
            by[h].append(img)
        for imgs in by.values():
            for img in imgs[1:]:
                issue(
                    size.source[img],
                    img,
                    "",
                    "중복·유사 이미지",
                    "기록만",
                    f"{imgs[0]}와 픽셀 동일",
                )
        # 유사: pHash 해밍 거리 ≤ 5 이면서 조합이 다른 경우 (같은 조합의 각도 차이는 정상)
        names = list(ph)
        H = np.array([ph[n] for n in names], dtype=np.uint64)
        grp = np.array([n.split("_")[0] for n in names])
        for s in range(0, len(H), 2000):
            dist = np.bitwise_count(H[s : s + 2000, None] ^ H[None, :])
            for i, j in zip(*np.nonzero(dist <= PHASH), strict=True):
                i += s
                if i < j and grp[i] != grp[j]:
                    issue(
                        size.source[names[j]],
                        names[j],
                        "",
                        "중복·유사 이미지",
                        "기록만",
                        f"{names[i]}와 pHash 거리 {dist[i - s, j]}",
                    )

    # ── 저장 ─────────────────────────────────────────────────────────────────
    iss = pd.DataFrame(
        issues,
        columns=["source", "image", "code", "check", "level", "action", "detail"],
    )
    META.mkdir(parents=True, exist_ok=True)
    save_table(iss, META / "label_issues.csv")
    lf = pd.DataFrame(files).merge(
        d.groupby(["source", "image", "code"]).img_id.first().reset_index(), how="left"
    )
    save_table(lf, META / "label_files.csv")

    ex_img = iss[iss.action == "이미지 제외"]
    ex_box = iss[iss.action == "bbox 제외"][["image", "code"]].drop_duplicates()
    n_combo = lf[
        lf.source == "combo"
    ].image.nunique()  # -오류 폴더만 있는 이미지도 센다
    n_kaggle = d[d.source == "kaggle"].image.nunique()

    lines = [
        "# 라벨 검수 결과 (분류기 crop 데이터)",
        "",
        f"- 실행일: {date.today()} / 스크립트: `python -m etl.classifier.validate_source_labels`"
        + (" (PNG 검사 생략)" if skip_png else ""),
        "- 기준 문서: Obsidian 「알약 데이터 검증 체크리스트」",
        f"- 대상: 조합 TL_{'·'.join(map(str, COMBO_ZIPS))} 중 촬영 조건(연회색·주백색·70/75/90°) 이미지 {n_combo}장, Kaggle train {n_kaggle}장, bbox {len(d)}개",
        f"- 기준값: 1mm당 픽셀 긴 변 {RATIO[0]}~{RATIO[1]}·짧은 변 {RATIO[0]} 이상, IoU > {IOU} 또는 작은 bbox의 {CONTAIN:.0%} 이상 겹침, pHash 거리 ≤ {PHASH}",
        f"- 실제 알약 크기 값(leng_long)이 0인 약: 같은 클래스 bbox 중앙값 대비 긴 변 {REL[0]}배·짧은 변 {REL[1]}배 이상이어야 정상",
        "- 처리: 조합은 라벨 오류가 하나라도 있으면 **이미지 제외**(대체 가능), Kaggle은 **bbox 제외**(대체 불가), crop 라벨에 영향 없는 항목은 **기록만**",
        "- 상세 목록: `data/classifier/meta/label_issues.csv` (문제 하나당 한 줄)",
        "",
        "## 항목별 결과",
        "",
        "| 구분 | 항목 | 성격 | 건수 | 이미지 수 | 이미지 제외 | bbox 제외 | 기록만 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c, part, lv in CHECKS:
        g = iss[iss.check == c]
        if skip_png and part == "Image" and c != "이미지 ID 중복":
            cnt = ["생략"] * 5
        else:
            cnt = [len(g), g[g.image != ""].image.nunique()] + [
                (g.action == a).sum() for a in ("이미지 제외", "bbox 제외", "기록만")
            ]
        lines.append(f"| {part} | {c} | {lv} | " + " | ".join(map(str, cnt)) + " |")
    lines += [
        "",
        "건수는 bbox 또는 이미지 단위 기록 수, 이미지 수는 관련된 서로 다른 이미지 수다.",
        "Annotation ID·이미지 ID 중복은 Kaggle만 검사했다. 조합 JSON은 약마다 파일이 따로라 id가 항상 1이다.",
        "조합 JSON의 category_id는 항상 1이고 dl_idx는 코드와 규칙이 일정하지 않아(대부분 코드-1, K-053384는 130376), 조합의 코드 매핑은 폴더 코드 = drug_N = dl_mapping_code 로 확인했다.",
        "",
        "## 적용 결과",
        "",
        "### 제외 방식: 조합은 이미지째, Kaggle은 bbox만",
        "",
        "분류기는 이미지 한 장이 아니라 **알약 하나(bbox 하나)를 잘라 낸 crop**으로 학습한다.",
        "그래서 한 이미지에서 라벨이 틀린 알약만 crop을 만들지 않고 건너뛰어도, 나머지 알약의 crop은 그대로 쓸 수 있다.",
        "",
        "| 출처 | 제외 단위 | 이유 |",
        "|---|---|---|",
        "| AI Hub 조합 | **이미지째** | 후보가 1만 장이 넘어 다른 이미지로 채울 수 있다. 한 bbox가 틀린 이미지는 다른 라벨도 의심스럽다 |",
        "| Kaggle | **틀린 bbox만** | 232장뿐이라 대신할 이미지가 없다. 이미지째 빼면 Kaggle에만 있는 약(하이트린, 기넥신 등)이 줄어든다 |",
        "",
        (
            "예) Kaggle `K-003351-016262-018357_…_75`: 알약 3개 중 K-018357 bbox가 이미지 밖(x=6567)이라 이 bbox만 빼고, "
            "K-003351·K-016262 crop 2장은 만든다."
        ),
        "",
        "### 제외 결과",
        "",
        f"- 조합: 이미지 **{ex_img[ex_img.source == 'combo'].image.nunique()}장** 제외 (그 이미지의 bbox는 모두 쓰지 않음)",
        (
            f"- Kaggle: 이미지 {ex_img[ex_img.source == 'kaggle'].image.nunique()}장 제외, "
            f"이미지 **{iss[iss.action == 'bbox 제외'].image.nunique()}장**에서 bbox **{len(ex_box)}개** 제외 (같은 이미지의 다른 bbox는 그대로 씀)"
        ),
        "",
        "### 전체 합계: 검수 대상 → 제외 → 남은 수",
        "",
        "| 출처 | 대상 이미지 | 대상 bbox | 제외 이미지 | 제외 bbox | 남은 이미지 | 남은 bbox |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    # 조합은 제외 이미지의 bbox 전부, Kaggle은 틀린 (이미지, 약) bbox만 빠진다
    drop = d.image.isin(ex_img.image) | d.set_index(["image", "code"]).index.isin(
        ex_box.set_index(["image", "code"]).index
    )
    tot = []
    for src, name, n_img in (
        ("combo", "AI Hub 조합", n_combo),
        ("kaggle", "Kaggle", n_kaggle),
    ):
        s = d.source == src
        row = [
            n_img,
            s.sum(),
            ex_img[ex_img.source == src].image.nunique(),
            (s & drop).sum(),
        ]
        row += [n_img - row[2], (s & ~drop).sum()]
        tot.append(row)
        lines.append(f"| {name} | " + " | ".join(f"{v:,}" for v in row) + " |")
    lines += [
        "| **합계** | "
        + " | ".join(f"**{sum(c):,}**" for c in zip(*tot, strict=True))
        + " |",
        "",
        "- 조합의 제외 bbox에는 같은 이미지에 있던 정상 알약 bbox도 들어 있다 (이미지째 빼므로).",
        "- AI Hub `-오류` 폴더 라벨은 읽지 않았으므로 대상·제외 bbox 수에 들어 있지 않다 (이미지 수에는 들어 있다).",
        "- 남은 수는 crop 후보이고, 실제로 쓴 이미지는 `select_source_images.py`가 이 중에서 고른다.",
        "",
        "### 제외 사유별 이미지 수",
        "",
        (
            "제외가 있었던 이미지 수다. 조합은 그 이미지를 통째로 뺀 수, Kaggle은 bbox를 뺀 이미지 수다. "
            "한 이미지가 여러 사유에 걸릴 수 있어 사유별 합은 위 결과보다 크다."
        ),
        "",
        "| 항목 | 조합 (이미지째 제외) | Kaggle (bbox만 제외한 이미지) |",
        "|---|---|---|",
    ]
    for c, *_ in CHECKS:
        g = iss[(iss.check == c) & iss.action.isin(["이미지 제외", "bbox 제외"])]
        if len(g):
            lines.append(
                f"| {c} | {g[g.source == 'combo'].image.nunique()} | {g[g.source == 'kaggle'].image.nunique()} |"
            )
    kb = iss[iss.action == "bbox 제외"]
    if len(kb):
        lines += [
            "",
            "### Kaggle에서 뺀 bbox 목록",
            "",
            "| 이미지 | 뺀 bbox | 사유 | 내용 |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {r.image[:-4]} | {r.code} | {r.check} | {r.detail} |"
            for r in kb.itertuples()
        ]
        lines += [
            "",
            (
                "같은 bbox에 두 약 라벨이 붙은 경우, 한쪽만 크기가 실제 알약과 맞으면 그쪽을 남기고 다른 쪽만 뺐다. "
                "어느 쪽이 맞는지 판단할 수 없으면 둘 다 뺐다."
            ),
        ]
    cls = iss[iss.source == "all"]
    if len(cls):
        lines += [
            "",
            "### 클래스 단위 기록",
            "",
            "| 항목 | 코드 | 내용 |",
            "|---|---|---|",
        ]
        lines += [f"| {r.check} | {r.code} | {r.detail} |" for r in cls.itertuples()]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n".join(lines[9 : 9 + len(CHECKS) + 2]))
    print(
        f"\n이미지 제외: 조합 {ex_img[ex_img.source == 'combo'].image.nunique()}장, Kaggle {ex_img[ex_img.source == 'kaggle'].image.nunique()}장 / bbox 제외 {len(ex_box)}개"
    )
    print("저장:", META / "label_issues.csv", REPORTS / "validation_report.md")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="PNG를 여는 검사(누락·손상·크기·중복 이미지)를 건너뛴다",
    )
    args = parser.parse_args(argv)
    validate_source_labels(skip_png=args.skip_png)


if __name__ == "__main__":
    main()
