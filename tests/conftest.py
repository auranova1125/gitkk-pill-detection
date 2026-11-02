"""여러 테스트 모듈에서 공유하는 합성 metadata fixture."""

from typing import Any

import pytest


@pytest.fixture
def classifier_class_map() -> dict[str, Any]:
    """3개 합성 품목의 분류 class map을 반환한다."""

    return {
        "num_classes": 3,
        "classes": [
            {
                "class_index": index,
                "category_id": 1900 + index,
                "item_seq": f"item-{index}",
                "name": f"drug-{index}",
            }
            for index in range(3)
        ],
    }


@pytest.fixture
def detector_label_map() -> dict[str, Any]:
    """background와 pill로 구성된 단일 클래스 label map을 반환한다."""

    return {
        "num_foreground_classes": 1,
        "num_model_classes": 2,
        "background_label": 0,
        "classes": [{"label": 1, "name": "pill"}],
    }
