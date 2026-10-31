"""통합 추론 이미지 DataLoader 테스트."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from data_loading.recognition import build_pill_recognition_inputs
from detector.inference import validate_inference_batch


def test_pill_recognition_loader_reads_sorted_variable_size_images(tmp_path) -> None:
    Image.new("RGB", (20, 12), color=(255, 0, 0)).save(tmp_path / "b.png")
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (16, 24), color=(0, 255, 0)).save(nested / "a.jpg")

    loader = build_pill_recognition_inputs(
        config=SimpleNamespace(input_dir=tmp_path),
        batch_size=2,
    )
    images, metadata = validate_inference_batch(next(iter(loader)))

    assert [tuple(image.shape) for image in images] == [(3, 12, 20), (3, 24, 16)]
    assert all(image.dtype == torch.float32 for image in images)
    assert [meta["file_name"] for meta in metadata] == ["b.png", "nested/a.jpg"]
    assert [meta["image_id"] for meta in metadata] == [0, 1]


def test_pill_recognition_loader_rejects_empty_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="contains no supported images"):
        build_pill_recognition_inputs(
            config=SimpleNamespace(input_dir=tmp_path),
            batch_size=1,
        )
