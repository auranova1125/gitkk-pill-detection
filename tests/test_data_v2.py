"""export_classifier_v2.apply_actions: v1 samples에 라벨 2차 검수 조치를 적용한다."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classifier_data_build"))
from export_classifier_v2 import apply_actions  # noqa: E402

CLASS_MAP = {"num_classes": 2, "classes": [
    {"class_index": 0, "category_id": 1, "item_seq": "100", "name": "가정"},
    {"class_index": 1, "category_id": 2, "item_seq": "200", "name": "나정"},
]}


def v1():
    rows = [("a_0", "a", "K-000001", "0"), ("a_1", "a", "K-000002", "1"),
            ("b_0", "b", "K-000001", "0"), ("b_1", "b", "K-000002", "1"),
            ("c_0", "c", "K-000001", "0")]
    return pd.DataFrame([dict(sample_id=s, image_id=i, code=c, class_index=k, category_id=str(int(c[2:])),
                              item_seq=str(100 * (int(k) + 1)), name="가정" if k == "0" else "나정",
                              crop_path=f"crops/{c}/{s}.png", split="val") for s, i, c, k in rows])


def actions(rows):
    return pd.DataFrame(rows, columns=["sample_id", "image_id", "code", "new_code", "action"])


def test_drops_and_swaps():
    act = actions([("a_0", "a", "K-000001", "", "이미지 제외"), ("a_1", "a", "K-000002", "", "이미지 제외"),
                   ("c_0", "c", "K-000001", "", "bbox 제외"),
                   ("b_0", "b", "K-000001", "K-000002", "라벨 교환 (수정)"),
                   ("b_1", "b", "K-000002", "K-000001", "라벨 교환 (수정)")])
    v2 = apply_actions(v1(), act, CLASS_MAP).set_index("sample_id")
    assert list(v2.index) == ["b_0", "b_1"]
    assert v2.loc["b_0", ["code", "class_index", "category_id", "item_seq", "name"]].tolist() == ["K-000002", "1", "2", "200", "나정"]
    assert v2.loc["b_0", "crop_path"] == "crops/K-000002/b_0.png"
    assert v2.loc["b_1", "code"] == "K-000001"


def test_rejects_partial_image_exclusion():
    act = actions([("a_0", "a", "K-000001", "", "이미지 제외")])          # a_1이 빠짐
    with pytest.raises(AssertionError, match="이미지 제외"):
        apply_actions(v1(), act, CLASS_MAP)


def test_rejects_unknown_sample_and_label_mismatch():
    with pytest.raises(AssertionError, match="v1에 없는"):
        apply_actions(v1(), actions([("z_0", "z", "K-000001", "", "bbox 제외")]), CLASS_MAP)
    with pytest.raises(AssertionError, match="다르다"):
        apply_actions(v1(), actions([("b_0", "b", "K-000002", "K-000001", "라벨 교환 (수정)")]), CLASS_MAP)
