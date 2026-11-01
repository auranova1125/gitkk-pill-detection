"""라벨 2차 검수 ④: label_actions.csv의 오류 사진을 탐지기 데이터 기준 목록으로 바꾼다.

같은 사진이 탐지기 학습 패키지에도 있으므로 탐지기 담당이 확인할 수 있게 넘긴다.
분류기 조치를 그대로 쓰면 안 되는 경우가 있어 탐지기용 권장 조치를 따로 적는다.
    분류기 이미지 제외   → 이미지 제외
    분류기 bbox 제외     → bbox 수정 또는 이미지 제외 (bbox만 지우면 그 알약을 배경으로 배운다)
    분류기 라벨 교환     → 라벨 교환 (탐지기도 118종 클래스를 쓴다)

사진은 file_name으로 찾는다 (탐지기 image_id는 패키지를 다시 만들면 바뀔 수 있다).

결과
    classifier_data_build/label_audit/detector_actions.csv   사진마다 한 줄

사용:
    python classifier_data_build/export_detector_audit.py [--package <training_package_detector.zip>]
"""
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

from config import AUDIT, DETECTOR_PACKAGE, save_table

DETECTOR_ACTION = {"이미지 제외": "이미지 제외", "bbox 제외": "bbox 수정 또는 이미지 제외",
                   "라벨 교환 (수정)": "라벨 교환"}
RANK = ["이미지 제외", "bbox 제외", "라벨 교환 (수정)"]

ap = argparse.ArgumentParser()
ap.add_argument("--package", type=Path, default=DETECTOR_PACKAGE)
args = ap.parse_args()

with zipfile.ZipFile(args.package) as z:
    manifest = pd.read_csv(io.BytesIO(z.read("split_manifest.csv")))
# 파일명 앞의 출처 표시(source_<n>_, codeit_)를 떼면 분류기 image_id와 같다
manifest["src_image_id"] = manifest.file_name.str.replace(r"^(source_\d+|codeit)_", "", regex=True).str.removesuffix(".png")

act = pd.read_csv(AUDIT / "label_actions.csv")
# 사진마다 가장 강한 조치 하나, 메모는 합친다
act["rank"] = act.action.map(RANK.index)
per_image = (act.sort_values("rank").groupby("image_id", as_index=False)
             .agg(분류기_조치=("action", "first"), source=("source", "first"),
                  분류기_split=("split", "first"), 출처=("출처", lambda x: ", ".join(sorted(set(x)))),
                  메모=("메모", lambda x: " / ".join(dict.fromkeys(x)))))
per_image["탐지기_권장_조치"] = per_image.분류기_조치.map(DETECTOR_ACTION)
# 사진 전체가 아니라 일부 알약만 고치면 그 알약 코드를 적는다
part = act[act.action != "이미지 제외"].groupby("image_id").code.agg(lambda x: ", ".join(x))
per_image["대상_알약"] = per_image.image_id.map(part).fillna("전체")
# 라벨 교환은 바꿀 두 알약을 적는다
swap = act[act.action == "라벨 교환 (수정)"]
for img, g in swap.groupby("image_id"):
    per_image.loc[per_image.image_id == img, "메모"] = ", ".join(f"{c} → {n}" for c, n in zip(g.code, g.new_code))

out = per_image.merge(manifest[["src_image_id", "file_name", "image_path", "dataset", "split", "group_id"]],
                      left_on="image_id", right_on="src_image_id", how="left")
missing = out.file_name.isna()
out = out[~missing].rename(columns={"split": "탐지기_split", "image_id": "원본_사진"})
out = out[["file_name", "image_path", "dataset", "탐지기_split", "group_id", "탐지기_권장_조치", "대상_알약",
           "분류기_조치", "출처", "메모", "원본_사진"]].sort_values(["탐지기_split", "file_name"])
save_table(out, AUDIT / "detector_actions.csv")

print(f"탐지기 데이터에 있는 오류 사진 {len(out)}장 / 전체 {len(per_image)}장")
print(out.groupby(["탐지기_split", "탐지기_권장_조치"]).size().to_string())
if missing.any():
    print(f"탐지기 데이터에 없는 사진 {missing.sum()}장 (분류기에만 있음):")
    for img in per_image.image_id[missing.values]:
        print(f"  {img}")
print(f"결과: {AUDIT / 'detector_actions.csv'}")
