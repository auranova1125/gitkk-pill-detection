"""Matplotlib 기반 결과 시각화에 필요한 공통 설정."""

import warnings
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.font_manager import FontProperties


KOREAN_FONT_FAMILIES = (
    "Apple SD Gothic Neo",
    "AppleGothic",
    "Malgun Gothic",
    "NanumGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
)


@lru_cache(maxsize=1)
def resolve_korean_font() -> "FontProperties | None":
    """현재 환경에서 사용할 수 있는 한글 폰트를 우선순위대로 찾는다."""

    from matplotlib import font_manager

    for family in KOREAN_FONT_FAMILIES:
        try:
            font_path = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return font_manager.FontProperties(fname=font_path)

    warnings.warn(
        "Korean font was not found. Install Apple SD Gothic Neo, Malgun Gothic, "
        "NanumGothic, or Noto Sans KR to render Korean labels.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None
