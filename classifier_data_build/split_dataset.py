"""crop을 탐지기 분할(split_manifest.csv)에 맞춰 train/val로 나누고 crops.csv에 split 열을 붙인다.

탐지기는 파일명의 알약 조합(group)을 통째로 train 또는 validation에 넣었다.
같은 조합 사진은 같은 알약 실물을 찍은 것이므로, 분류기도 그룹을 탐지기와 같은 쪽에 넣는다.
그래야 탐지기 val 사진의 알약이 분류기 train에 들어가지 않고, 탐지→분류 전체를 같은 val로 평가할 수 있다.
탐지기 목록에 없는 그룹은 train에 넣는다 (분류기 val ⊂ 탐지기 validation 그룹).

사용:
    python classifier_data_build/split_dataset.py [--package data/detector/training_package_detector.zip]
"""
import argparse
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from config import CLASSES, CROPS, DETECTOR_PACKAGE, META, REPORTS, save_table

ap = argparse.ArgumentParser()
ap.add_argument("--package", type=Path, default=DETECTOR_PACKAGE,
                help="탐지기 학습 패키지 (split_manifest.csv가 들어 있음)")
args = ap.parse_args()

z = zipfile.ZipFile(args.package)
# 패키지 버전에 따라 맨 위에 있거나 training_package/ 아래에 있다
manifest = [n for n in z.namelist() if n.split("/")[-1] == "split_manifest.csv"]
assert len(manifest) == 1, f"split_manifest.csv를 하나만 찾아야 함 ({manifest})"
man = pd.read_csv(io.BytesIO(z.read(manifest[0])))
assert man.groupby("group_id").split.nunique().max() == 1, "탐지기 분할에 train/val에 걸친 그룹이 있다"
group_split = man.groupby("group_id").split.first().map({"train": "train", "validation": "val"})

df = pd.read_csv(CROPS / "crops.csv")
df["split"] = df.group.map(group_split)
new = df.split.isna()
df["split_from"] = "탐지기 분할"
df.loc[new, "split_from"] = "탐지기 목록에 없는 그룹 → train"
df["split"] = df.split.fillna("train")

t = df.pivot_table(index="code", columns="split", values="crop", aggfunc="count", fill_value=0)
print(f"train {t.train.sum()}개, val {t.val.sum()}개 (val {t.val.sum() / len(df):.1%}), "
      f"그룹 train {df[df.split == 'train'].group.nunique()}개 / val {df[df.split == 'val'].group.nunique()}개")
print(f"탐지기 목록에 없는 그룹: {df[new].group.nunique()}개, crop {new.sum()}개 → train")
print(f"train/val 양쪽에 걸친 그룹: {(df.groupby('group').split.nunique() > 1).sum()}개")
print(f"val이 없는 클래스 {(t.val == 0).sum()}종: {t[t.val == 0].index.tolist()}")

# 그룹은 달라도 거의 같은 사진(pHash 거리 ≤ 5)이 train과 val에 나뉜 경우 (validate_labels.py 기록)
iss = pd.read_csv(META / "label_issues.csv", keep_default_na=False)
sim = iss[iss.check == "중복·유사 이미지"]
other = sim.detail.str.extract(r"^(.+?)와 pHash")[0]
img_split = df.drop_duplicates("image").set_index("image").split
pairs = pd.DataFrame({"a": sim.image, "b": other, "sa": sim.image.map(img_split), "sb": other.map(img_split)}).dropna()
cross = pairs[pairs.sa != pairs.sb]
leak_img = set(cross.a[cross.sa == "val"]) | set(cross.b[cross.sb == "val"])   # train에 거의 같은 사진이 있는 val 이미지
val_img = set(df[df.split == "val"].image)
leak_crop = df[df.image.isin(leak_img)]
# 평가 때 이 열이 True인 val crop을 빼고도 점수를 본다 (train 사진을 외워서 맞힌 몫을 확인)
df["near_dup_in_train"] = df.image.isin(leak_img)
save_table(df, CROPS / "crops.csv")
print(f"유사 이미지 쌍 중 train/val에 나뉜 쌍: {len(cross)}개 / {len(pairs)}개 (탐지기 분할에도 똑같이 있음)")
print(f"train에 거의 같은 사진이 있는 val 이미지: {len(leak_img)}장 / {len(val_img)}장, crop {len(leak_crop)}개")

# ── 기록용 표 (classifier_data_build/reports/split_report.md) ──────────────────
cls = pd.read_csv(CLASSES).set_index("code")
t["total"] = t.train + t.val
t["val_pct"] = t.val / t.total * 100
t["leak"] = leak_crop.code.value_counts().reindex(t.index, fill_value=0)
t = t.join(cls[["class_id", "name"]]).sort_values("class_id")
by_src = df.pivot_table(index="source", columns="split", values="crop", aggfunc="count", fill_value=0)


def row(code, r):
    return (f"| {r.class_id} | {code} | {r['name']} | {r.total} | {r.train} | {r.val} | {r.val_pct:.1f}% | {r.leak} |")


head = ["| class_id | 약 코드 | 품목명 | 합계 | train | val | val 비율 | val 중 train에 유사 사진 |",
        "|---:|---|---|---:|---:|---:|---:|---:|"]
md = [
    "# train/val 분할 결과 (분류기 crop)", "",
    f"`python classifier_data_build/split_dataset.py` 결과 ({date.today()}). `data/classifier/crops/crops.csv`의 `split` 열 기준이다.", "",
    "## 분할 기준", "",
    f"- 탐지기 학습 패키지 `{args.package.name}`의 `split_manifest.csv`를 그대로 따른다.",
    "- 그룹 = 파일명의 알약 코드 조합 (예: `K-001900-016548-019607-029451`). 같은 그룹은 탐지기와 같은 쪽(train 또는 val)에 넣는다.",
    "- 탐지기 목록에 없는 그룹은 train에 넣는다. 그래서 분류기 val은 모두 탐지기 validation 그룹 안에 있다.", "",
    "## 요약", "",
    "| 항목 | train | val |", "|---|---:|---:|",
    f"| **crop** | **{t.train.sum():,}** | **{t.val.sum():,}** ({t.val.sum() / len(df):.1%}) |",
]
md += [f"| └ {'AI Hub 조합' if k == 'combo' else 'Kaggle'} | {v.train:,} | {v.val:,} |" for k, v in by_src.iterrows()]
md += [
    f"| 그룹 | {df[df.split == 'train'].group.nunique():,} | {df[df.split == 'val'].group.nunique():,} |",
    f"| 클래스별 최소 / 중앙값 / 최대 | {t.train.min()} / {t.train.median():.0f} / {t.train.max()} | {t.val.min()} / {t.val.median():.0f} / {t.val.max()} |", "",
    "| 확인 | 결과 |", "|---|---|",
    f"| train/val 양쪽에 걸친 그룹 | {(df.groupby('group').split.nunique() > 1).sum()}개 |",
    f"| val이 없는 클래스 | {(t.val == 0).sum()}종 |",
    f"| 클래스별 val 비율 | {t.val_pct.min():.1f}% ~ {t.val_pct.max():.1f}% (중앙값 {t.val_pct.median():.1f}%) |",
    f"| 탐지기 목록에 없어 train에 넣은 그룹 | {df[new].group.nunique()}개 (crop {new.sum()}개) |",
    f"| 그룹은 다르지만 거의 같은 사진(pHash 거리 ≤ 5)이 train/val에 나뉜 쌍 | {len(cross)}쌍 |",
    f"| train에 거의 같은 사진이 있는 val 이미지 | {len(leak_img)}장 / {len(val_img)}장 (crop {len(leak_crop)}개) |", "",
    "거의 같은 사진은 대부분 알약 하나만 다른 조합이다 (나머지 알약은 같은 실물, 같은 자리). "
    "탐지기 분할에도 똑같이 있으며, 이 val crop은 점수가 실제보다 높게 나올 수 있다.", "",
    f"`crops.csv`의 `near_dup_in_train` 열이 True인 crop이 이 {len(leak_crop)}개다. 분류기를 평가할 때 "
    f"**전체 val({t.val.sum():,}개)**과 **이 열이 False인 val({t.val.sum() - len(leak_crop):,}개)** 점수를 함께 본다. "
    "차이가 크면 train 사진을 외워서 맞힌 몫이 크다는 뜻이다.", "",
    "```python",
    "df = pd.read_csv(CROPS / \"crops.csv\")",
    "val_all = df[df.split == \"val\"]",
    "val_clean = val_all[~val_all.near_dup_in_train]",
    "```", "",
    "## 클래스별 (class_id 순)", "", *head, *[row(c, r) for c, r in t.iterrows()], "",
    "## val이 적은 순 (상위 15)", "", *head, *[row(c, r) for c, r in t.sort_values(["val", "class_id"]).head(15).iterrows()],
]
REPORTS.mkdir(exist_ok=True)
(REPORTS / "split_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("저장:", CROPS / "crops.csv", REPORTS / "split_report.md")
