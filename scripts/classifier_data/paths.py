"""분류기 데이터 구성 스크립트에서 공유하는 경로와 원본 zip 탐색."""

import os
import re
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"  # git에 올리지 않음 (.gitignore의 /data/)
# 원본 위치. 다른 곳에 있으면 환경 변수로 지정하거나 data/ 아래에 심볼릭 링크를 둔다
AIHUB = Path(
    os.environ.get("AIHUB_DIR", DATA / "aihub")
).expanduser()  # AI Hub 원본 zip (TS: 이미지, TL: 라벨)
KAGGLE = Path(
    os.environ.get("KAGGLE_DIR", DATA / "kaggle")
).expanduser()  # train_images/, train_annotations/
CLASSES = (
    ROOT / "scripts" / "classifier_data" / "classes.csv"
)  # class_id, code, category_id, name (118종)
META = DATA / "classifier" / "meta"  # 선정 목록, bbox
CROPS = DATA / "classifier" / "crops"  # crops/<약 코드>/*.jpg
EXPORT = DATA / "classifier" / "export"  # 분류기 팀에 넘기는 입력 패키지
DETECTOR_PACKAGE = (
    DATA / "detector" / "training_package_detector.zip"
)  # 탐지기 학습 패키지 (split_manifest.csv)
REPORTS = (
    ROOT / "reports" / "data" / "classifier"
)  # 실행 결과 문서 (자동 생성, git에 올림)

COMBO_ZIPS = [
    1,
    3,
    4,
    5,
    6,
    7,
    8,
]  # 2번 세트(TS_2 이미지·TL_2 라벨)는 test와 겹칠 수 있어 쓰지 않는다


def save_table(df, path, index=False):
    """CSV로 저장한다. utf-8-sig(BOM)라 엑셀에서 열어도 한글이 덜 깨진다."""
    df.to_csv(path, index=index, encoding="utf-8-sig")


def _nfc(s):
    return unicodedata.normalize("NFC", s)  # macOS 파일명은 한글이 NFD로 저장된다


def ts_zip(n):
    """TS_{n}_조합 이미지 zip 경로. AIHUB 아래 어디에 있어도 찾는다."""
    hits = [p for p in AIHUB.rglob("*.zip") if re.match(rf"TS_{n}_조합", _nfc(p.name))]
    assert len(hits) == 1, f"TS_{n}_조합 zip을 찾지 못함 ({hits})"
    return hits[0]


def tl_zip(n):
    """TL_{n}_조합 라벨 zip을 연다."""
    hits = [p for p in AIHUB.rglob("*.zip") if re.match(rf"TL_{n}_조합", _nfc(p.name))]
    assert len(hits) == 1, f"TL_{n}_조합 zip을 찾지 못함 ({hits})"
    return zipfile.ZipFile(hits[0])
