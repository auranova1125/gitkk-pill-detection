"""AI Hub val 외부 평가 패키지 생성의 단계별 규칙을 검사한다."""

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from etl.classifier.aihub_val.build_eval_package import (
    assign_sample_ids,
    build_samples,
    check_conditions,
    find_training_duplicates,
    read_review_records,
    read_val_labels,
)

GROUP = "K-001900-016548-019607"


def _class_map() -> dict[str, Any]:
    return {
        "num_classes": 3,
        "classes": [
            {"class_index": 0, "category_id": 1900, "item_seq": "100", "name": "가정"},
            {"class_index": 1, "category_id": 16548, "item_seq": "200", "name": "나정"},
            {"class_index": 2, "category_id": 19607, "item_seq": "300", "name": "다정"},
        ],
    }


def _label_json(stem: str, code: str, item_seq: int, **image: Any) -> bytes:
    info = {
        "file_name": f"{stem}.png",
        "item_seq": item_seq,
        "drug_dir": "앞면",
        "back_color": "연회색 배경",
        "light_color": "주백색",
        "camera_la": int(stem.split("_")[5]),
        **image,
    }
    return json.dumps(
        {"images": [info], "annotations": [{"bbox": [10, 20, 30, 40]}]}
    ).encode()


def _write_zips(
    tmp_path: Path, *, back_color: str = "연회색 배경"
) -> tuple[Path, Path]:
    label_zip, image_zip = tmp_path / "VL_1_조합.zip", tmp_path / "VS_1_조합.zip"
    seqs = {"K-001900": 100, "K-016548": 200, "K-019607": 300}
    with (
        zipfile.ZipFile(label_zip, "w") as labels,
        zipfile.ZipFile(image_zip, "w") as images,
    ):
        for angle in ("70", "90"):
            stem = f"{GROUP}_0_2_0_2_{angle}_000_200"
            for code, seq in seqs.items():
                extra = {"back_color": back_color} if angle == "90" else {}
                labels.writestr(
                    f"{GROUP}_json/{code}/{stem}.json",
                    _label_json(stem, code, seq, **extra),
                )
            path = tmp_path / f"{stem}.png"
            Image.new("RGB", (64, 80), color=(120, 120, 120)).save(path)
            images.write(path, f"{GROUP}/{stem}.png")
        index_path = tmp_path / "index.png"
        Image.new("RGB", (8, 8)).save(index_path)
        images.write(index_path, f"{GROUP}/{GROUP}_index.png")
    return label_zip, image_zip


def test_labels_get_stable_sample_ids_from_file_name_order(tmp_path: Path) -> None:
    label_zip, _ = _write_zips(tmp_path)

    labels = assign_sample_ids(read_val_labels(label_zip))

    assert len(labels) == 6
    first = labels[labels.image_id.str.contains("_70_")].set_index("code").sample_id
    assert first["K-001900"].endswith("_70_000_200_0")
    assert first["K-019607"].endswith("_70_000_200_2")
    assert set(labels.group_id) == {GROUP}


def test_check_conditions_flags_background_and_unknown_class(tmp_path: Path) -> None:
    label_zip, _ = _write_zips(tmp_path, back_color="흰색 배경")
    labels = assign_sample_ids(read_val_labels(label_zip))
    class_map = _class_map()
    class_map["classes"] = class_map["classes"][:2]  # K-019607은 class_map 밖

    excluded = check_conditions(labels, class_map)

    background = excluded[excluded.reason == "촬영 조건 다름"]
    assert set(background.image_id) == {f"{GROUP}_0_2_0_2_90_000_200"}
    assert set(excluded[excluded.reason == "118종 밖 클래스"].code) == {"K-019607"}


def test_build_samples_uses_class_map_and_eval_split(tmp_path: Path) -> None:
    label_zip, image_zip = _write_zips(tmp_path)
    labels = assign_sample_ids(read_val_labels(label_zip))

    samples = build_samples(labels, _class_map(), image_zip)

    assert set(samples.split) == {"eval"}
    assert samples.set_index("code").class_index.groupby(level=0).first().to_dict() == {
        "K-001900": 0,
        "K-016548": 1,
        "K-019607": 2,
    }
    assert samples.image_path.iloc[0].startswith(f"aihub_val/VS_1_조합.zip/{GROUP}/")
    assert (
        samples.crop_path == "crops/" + samples.code + "/" + samples.sample_id + ".png"
    ).all()


def test_build_samples_rejects_item_seq_mismatch(tmp_path: Path) -> None:
    label_zip, image_zip = _write_zips(tmp_path)
    labels = assign_sample_ids(read_val_labels(label_zip))
    class_map = _class_map()
    class_map["classes"][0]["item_seq"] = "999"

    with pytest.raises(ValueError, match="item_seq"):
        build_samples(labels, class_map, image_zip)


def test_training_duplicates_cover_exact_and_near_images() -> None:
    eval_hashes = pd.DataFrame(
        {
            "image_id": ["same", "near", "far"],
            "md5": ["m1", "e2", "e3"],
            "phash": np.array([0, 0b111, 2**64 - 1], dtype=np.uint64),
        }
    )
    train_hashes = pd.DataFrame(
        {
            "image_id": ["t1", "t2"],
            "split": ["train", "val"],
            "md5": ["m1", "t2"],
            "phash": np.array([2**40, 0], dtype=np.uint64),
        }
    )

    duplicates = find_training_duplicates(eval_hashes, train_hashes, max_distance=5)

    rows = duplicates.set_index("image_id")
    assert rows.loc["same", "kind"] == "픽셀 동일"
    assert rows.loc["same", "train_image_id"] == "t1"
    assert rows.loc["near", "train_image_id"] == "t2"
    assert rows.loc["near", "phash_distance"] == 3
    assert "far" not in rows.index


def test_review_records_are_optional(tmp_path: Path) -> None:
    assert read_review_records(tmp_path) is None

    pd.DataFrame(
        [
            {
                "sample_id": "a_0",
                "image_id": "a",
                "code": "K-001900",
                "판정": "정상",
                "조치": "",
                "메모": "",
                "검수": "사람 확인 완료",
            }
        ]
    ).to_csv(tmp_path / "position_suspects.csv", index=False)
    frames = read_review_records(tmp_path)

    assert frames is not None
    assert len(frames["position_suspects.csv"]) == 1
    assert frames["class_suspects.csv"].empty
    assert "suspect_code" in frames["class_suspects.csv"].columns
