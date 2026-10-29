"""data/classifier/crops 의 crop을 클래스별로 세고 data/classifier/crops/crop_counts.csv 로 저장한다.
개수는 폴더에 실제로 있는 파일 기준, 출처와 split별 개수는 crops.csv 기준.

사용:
    python -m scripts.classifier_data.count_crops [--min 108]
"""

import argparse

import pandas as pd

from scripts.classifier_data.config import CLASSES, CROPS, REPORTS, save_table

ap = argparse.ArgumentParser()
ap.add_argument(
    "--min", type=int, default=108, help="목표 개수 (이보다 적은 클래스를 따로 셈)"
)
args = ap.parse_args()

classes = pd.read_csv(CLASSES).set_index("code")
df = pd.read_csv(CROPS / "crops.csv")

t = classes[["class_id"]].copy()
t["name"] = classes.name
t["total"] = [len(list((CROPS / c).glob("*.jpg"))) for c in t.index]
for col in ("source", "split"):
    if col in df:
        by = df.pivot_table(
            index="code", columns=col, values="crop", aggfunc="count", fill_value=0
        )
        for k in by.columns:
            t[k] = by[k].reindex(t.index, fill_value=0)
t = t.sort_values("total")

pd.set_option(
    "display.max_rows",
    None,
    "display.width",
    200,
    "display.unicode.east_asian_width",
    True,
)
print(
    t.assign(name=t.name.str.slice(0, 20)).to_string()
)  # 화면에서는 긴 이름을 잘라 표가 넓어지지 않게
print()
print(f"클래스 {len(t)}종, crop 합계 {t.total.sum()}개")
print(
    f"클래스별 최소 {t.total.min()}, 중앙값 {t.total.median():.0f}, 최대 {t.total.max()}"
)
print(f"{args.min}개 미만 {(t.total < args.min).sum()}종")
if "val" in t:
    print(f"클래스별 val 최소 {t.val.min()}, 중앙값 {t.val.median():.0f}")

save_table(t, CROPS / "crop_counts.csv", index=True)

# 기록용 표는 reports/data/classifier/에 저장한다. 클래스 번호 순, 목표 미만은 굵게 표시한다.
from datetime import date

head_src = {"combo": "AI Hub 조합", "kaggle": "Kaggle"}
md = [
    "# 클래스별 crop 개수",
    "",
    f"`data/classifier/crops/<약 코드>/` 폴더의 jpg 개수 ({date.today()}, `python -m scripts.classifier_data.count_crops` 결과).",
    f"출처(Kaggle/조합)는 `data/classifier/crops/crops.csv` 기준이다. 목표 {args.min}개 미만은 **굵게** 표시.",
    "",
    "## 요약",
    "",
    "| 항목 | 값 |",
    "|---|---:|",
    f"| **crop 총 개수** | **{t.total.sum():,}장** |",
]
for k, v in df.source.value_counts().items():
    md.append(f"| └ {head_src.get(k, k)} | {v:,}장 |")
md += [
    f"| 원본 이미지 | {df.image.nunique():,}장 |",
    f"| 클래스 | {len(t)}종 |",
    f"| 클래스별 최소 / 중앙값 / 최대 | {t.total.min()} / {t.total.median():.0f} / {t.total.max()} |",
    f"| 목표 {args.min}개 미만 / 정확히 {args.min}개 / 초과 | {(t.total < args.min).sum()}종 / {(t.total == args.min).sum()}종 / {(t.total > args.min).sum()}종 |",
    "",
    "## 클래스별",
    "",
]
extra = [
    c for c in ("kaggle", "combo", "train", "val") if c in t
]  # train/val은 분할한 뒤에만 있다
head = {"kaggle": "Kaggle", "combo": "조합", "train": "train", "val": "val"}


def table(rows):
    out = [
        "| class_id | 약 코드 | 품목명 | 합계 | "
        + " | ".join(head[c] for c in extra)
        + " |",
        "|---:|---|---|---:|" + "---:|" * len(extra),
    ]
    for code, r in rows.iterrows():
        n = f"**{r.total}**" if r.total < args.min else str(r.total)
        out.append(
            f"| {r.class_id} | {code} | {r['name']} | {n} | "
            + " | ".join(str(r[c]) for c in extra)
            + " |"
        )
    return out


md += ["### 클래스 번호 순", ""] + table(t.sort_values("class_id"))
md += ["", "### 개수 적은 순", ""] + table(t.sort_values(["total", "class_id"]))
REPORTS.mkdir(exist_ok=True)
(REPORTS / "crop_counts.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("저장:", CROPS / "crop_counts.csv", REPORTS / "crop_counts.md")
