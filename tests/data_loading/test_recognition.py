"""통합 추론 이미지 DataLoader 테스트."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from data_loading.recognition import build_pill_recognition_inputs
from detector.inference import validate_inference_batch


def test_pill_recognition_loader_reads_sorted_variable_size_images(tmp_path) -> None:
    Image.new("RGB", (20, 12), color=(255, 0, 0)).save(tmp_path / "10.png")
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (16, 24), color=(0, 255, 0)).save(nested / "2.jpg")

    loader = build_pill_recognition_inputs(
        config=SimpleNamespace(input_dir=tmp_path),
        batch_size=2,
    )
    images, metadata = validate_inference_batch(next(iter(loader)))

    assert [tuple(image.shape) for image in images] == [(3, 24, 16), (3, 12, 20)]
    assert all(image.dtype == torch.float32 for image in images)
    assert [meta["file_name"] for meta in metadata] == ["nested/2.jpg", "10.png"]
    assert [meta["image_id"] for meta in metadata] == [2, 10]


def test_pill_recognition_loader_rejects_empty_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="contains no supported images"):
        build_pill_recognition_inputs(
            config=SimpleNamespace(input_dir=tmp_path),
            batch_size=1,
        )


def test_pill_recognition_loader_rejects_non_numeric_filename(tmp_path) -> None:
    Image.new("RGB", (8, 8)).save(tmp_path / "image.png")

    with pytest.raises(ValueError, match="filename stem must be an ASCII integer"):
        build_pill_recognition_inputs(
            config=SimpleNamespace(input_dir=tmp_path),
            batch_size=1,
        )


def test_pill_recognition_loader_rejects_duplicate_filename_ids(tmp_path) -> None:
    Image.new("RGB", (8, 8)).save(tmp_path / "1.png")
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (8, 8)).save(nested / "1.jpg")

    with pytest.raises(ValueError, match="duplicate pill recognition image_id"):
        build_pill_recognition_inputs(
            config=SimpleNamespace(input_dir=tmp_path),
            batch_size=1,
        )
