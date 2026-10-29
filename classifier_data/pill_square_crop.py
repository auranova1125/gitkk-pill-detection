"""기존 분류기 데이터 코드가 사용하는 crop 함수 호환 모듈."""

from pilldata.transforms import CROP_MARGIN as MARGIN
from pilldata.transforms import PADDING_COLOR_RGB as PAD_COLOR
from pilldata.transforms import crop_pill, cut_pill, pad_square

__all__ = [
    "MARGIN",
    "PAD_COLOR",
    "crop_pill",
    "cut_pill",
    "pad_square",
]
