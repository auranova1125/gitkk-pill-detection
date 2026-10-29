"""선정된 이미지(data/classifier/meta)의 bbox대로 알약을 잘라 클래스 폴더에 저장한다.
조합 이미지는 TS zip에서 바로 읽으므로 원본 이미지를 풀어 둘 필요가 없다.

결과 (ImageFolder 형식):
    data/classifier/crops/<약 코드>/<이미지>_<번호>.jpg
    data/classifier/crops/crops.csv     crop, image, code, source, group, x, y, w, h (원본 이미지 기준 bbox)

자르는 방법은 pilldata.transforms.crop_pill을 사용한다. bbox 사방 5% 여백을 더하고 이미지 경계에서 clip한 뒤 고정색 padding으로 정사각형을 만든다.
라벨 오류 bbox는 validate_labels.py → select_images.py에서 이미 빠졌으므로 여기서는 거르지 않는다.
jpg는 개수 세기·노트북 확인용이다. 학습에 쓰는 PNG crop(padding 전)은 export_classifier_inputs.py가 원본에서 다시 자른다.

사용:
    python -m scripts.classifier_data.make_crops
"""

import io
import os
import shutil
import zipfile

import pandas as pd
from PIL import Image

from pilldata.transforms import crop_pill
from scripts.classifier_data.config import (
    COMBO_ZIPS,
    CROPS,
    KAGGLE,
    META,
    save_table,
    ts_zip,
)

sel = pd.read_csv(META / "selected_images.csv")
boxes = pd.read_csv(META / "boxes.csv").groupby("image")
# 파일명 -> (zip, zip 안 경로). TL과 TS의 zip 번호가 다른 이미지가 있어(6장) 파일명으로 찾는다
where = {}
for n in COMBO_ZIPS:
    z = zipfile.ZipFile(ts_zip(n))
    for m in z.namelist():
        if m.endswith(".png"):
            where[os.path.basename(m)] = (z, m)


def load(r):
    if r.source == "kaggle":
        return Image.open(KAGGLE / "train_images" / r.image)
    z, m = where[r.image]
    return Image.open(io.BytesIO(z.read(m)))


if CROPS.exists():
    shutil.rmtree(CROPS)
rows = []
for k, r in enumerate(sel.itertuples(), 1):
    im = load(r).convert("RGB")
    for i, b in enumerate(boxes.get_group(r.image).itertuples()):
        (CROPS / b.code).mkdir(parents=True, exist_ok=True)
        name = f"{r.image[:-4]}_{i}.jpg"
        crop_pill(im, (b.x, b.y, b.w, b.h)).save(CROPS / b.code / name, quality=95)
        rows.append(
            (f"{b.code}/{name}", r.image, b.code, r.source, r.group, b.x, b.y, b.w, b.h)
        )
    if k % 500 == 0:
        print(f"  {k}/{len(sel)}장")

df = pd.DataFrame(
    rows, columns=["crop", "image", "code", "source", "group", "x", "y", "w", "h"]
)
save_table(df, CROPS / "crops.csv")
print(f"crop {len(df)}개 / 클래스 {df.code.nunique()}종")
