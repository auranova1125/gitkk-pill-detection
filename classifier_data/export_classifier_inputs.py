"""분류기 팀에 넘길 입력 데이터를 만든다 (구글 드라이브 업로드용 폴더 + zip). 데이터 파일만 넣는다.

「분류기 데이터 및 함수 입력 명세서」 + 수정 사항 1·2를 따른다.
    1. bbox 사방 5% 여백을 더한 뒤 이미지 경계로 clip  (crop.py의 cut_pill)
    2. 탐지 결과도 같은 방식으로 자른다                  (crop.py의 crop_pill)

결과: data/classifier/export/
    training_package_classifier/
        class_map.json
        data_summary.json
        samples.csv                 샘플(알약 하나) 하나당 한 줄
        crops/<약 코드>/<sample_id>.png
    training_package_classifier.zip   위 폴더의 내용 (zip 맨 위에 바로). 드라이브: gitkk_archive/training_package_classifier.zip

로더 코드(classifier_inputs.py, crop.py)는 패키지에 넣지 않는다. 레포에서 가져다 쓴다.

앞 단계: validate_labels → select_images → make_crops → split_dataset
사용:
    python classifier_data/export_classifier_inputs.py
"""
import io
import json
import os
import shutil
import sys
import zipfile
from types import SimpleNamespace

import pandas as pd
import torch
from PIL import Image

from config import CLASSES, COMBO_ZIPS, CROPS, EXPORT, KAGGLE, META, ROOT, tl_zip, ts_zip
from crop import MARGIN, cut_pill

sys.path.insert(0, str(ROOT))   # classifier_inputs.py가 classifier_data.crop을 import한다
from classifier_data import classifier_inputs as ci  # noqa: E402

OUT = EXPORT
PKG = OUT / "training_package_classifier"

crops = pd.read_csv(CROPS / "crops.csv")
assert "split" in crops, "split_dataset.py를 먼저 실행하세요"
files = pd.read_csv(META / "label_files.csv")
files["code"] = files.code.str.replace("-오류", "", regex=False)
classes = pd.read_csv(CLASSES)
to_index = dict(zip(classes.code, classes.class_id))

# ── 라벨 JSON에서 item_seq, drug_dir 읽기 ─────────────────────────────────
tl = {os.path.basename(z.filename): z for z in (tl_zip(n) for n in COMBO_ZIPS)}
crops = crops.merge(files[["image", "code", "json"]], on=["image", "code"], how="left")
assert crops.json.notna().all()


def read_json(p):
    if p.startswith("kaggle/"):
        return json.loads((KAGGLE.parent / p).read_text(encoding="utf-8"))
    zname, member = p.split("/", 1)
    return json.loads(tl[zname].read(member))


info = [read_json(p)["images"][0] for p in crops.json]
crops["item_seq"] = [str(i["item_seq"]) for i in info]
crops["drug_dir"] = [i.get("drug_dir") or "" for i in info]
seq = crops.groupby("code").item_seq.unique()
assert (seq.map(len) == 1).all(), f"item_seq가 여러 개인 클래스: {seq[seq.map(len) > 1].to_dict()}"

# ── 원본 사진 위치 (조합은 TS zip 안) ─────────────────────────────────────
where = {}
for n in COMBO_ZIPS:
    z = zipfile.ZipFile(ts_zip(n))
    for m in z.namelist():
        if m.endswith(".png"):
            where[os.path.basename(m)] = (z, m)


def image_path(r):
    if r.source == "kaggle":
        return f"kaggle/train_images/{r.image}"
    z, m = where[r.image]
    return f"aihub/{os.path.basename(z.filename)}/{m}"


# ── samples.csv ──────────────────────────────────────────────────────────
s = pd.DataFrame({
    "sample_id": crops.crop.str.split("/").str[-1].str[:-4],      # <원본 파일명>_<번호>: annotation 단위로 고유
    "split": crops.split,
    "class_index": crops.code.map(to_index),
    "code": crops.code,
    "category_id": crops.code.str[2:].astype(int),
    "item_seq": crops.item_seq,
    "name": crops.code.map(dict(zip(classes.code, classes.name))),
    "image_id": crops.image.str[:-4],
    "image_path": [image_path(r) for r in crops.itertuples()],
    "bbox_xywh": [json.dumps([float(v) for v in b]) for b in crops[["x", "y", "w", "h"]].values],
    "group_id": crops.group,
    "drug_dir": crops.drug_dir,
    "source": crops.source.map({"combo": "aihub", "kaggle": "kaggle"}),
    "near_dup_in_train": crops.near_dup_in_train,
})
s["crop_path"] = "crops/" + s.code + "/" + s.sample_id + ".png"
assert s.sample_id.is_unique

# ── crop PNG: 원본에서 여백 5% + clip으로 자른 직사각형 (padding·resize는 로더가 한다) ──
if PKG.exists():
    shutil.rmtree(PKG)
PKG.mkdir(parents=True)
for k, (img, g) in enumerate(s.groupby("image_id", sort=False), 1):
    src = crops[crops.image == img + ".png"].iloc[0]
    if src.source == "kaggle":
        im = Image.open(KAGGLE / "train_images" / src.image)
    else:
        z, m = where[src.image]
        im = Image.open(io.BytesIO(z.read(m)))
    im = im.convert("RGB")
    for r in g.itertuples():
        out = PKG / r.crop_path
        out.parent.mkdir(parents=True, exist_ok=True)
        cut_pill(im, json.loads(r.bbox_xywh), MARGIN).save(out)
    if k % 500 == 0:
        print(f"  {k}/{s.image_id.nunique()}장")

s.to_csv(PKG / "samples.csv", index=False, encoding="utf-8")

# ── class_map.json, data_summary.json ────────────────────────────────────
first = s.drop_duplicates("class_index").sort_values("class_index")
class_map = {"num_classes": len(classes), "classes": [
    {"class_index": int(r.class_index), "category_id": int(r.category_id), "item_seq": r.item_seq, "name": r.name}
    for r in first.itertuples()]}
assert len(class_map["classes"]) == len(classes)
(PKG / "class_map.json").write_text(json.dumps(class_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
summary = ci.make_data_summary(ci.read_samples(PKG), len(classes))
(PKG / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ── 검증 후 zip ──────────────────────────────────────────────────────────
cfg = SimpleNamespace(batch_size=32, seed=42, class_map_path=PKG / "class_map.json", data_summary_path=PKG / "data_summary.json")
loaders = ci.build_classifier_inputs(config=cfg, device=torch.device("cpu"))
ci.check_inputs(*loaders, expected_num_classes=len(classes))
print(f"입력 검증 완료: train {summary['num_train_samples']}, val {summary['num_val_samples']}")

zip_path = shutil.make_archive(str(OUT / PKG.name), "zip", PKG)
print("저장:", PKG, zip_path, f"{os.path.getsize(zip_path) / 1e6:.0f}MB")
