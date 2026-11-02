"""분류 학습 패키지 ETL 조립 테스트."""

from pathlib import Path

import pytest

import etl.classifier.build_classifier_dataset as etl_module


def test_classifier_etl_runs_explicit_stages_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    detector_package = Path("detector-package.zip")

    monkeypatch.setattr(
        etl_module,
        "validate_source_labels",
        lambda *, skip_png: calls.append(("validate", skip_png)),
    )
    monkeypatch.setattr(
        etl_module,
        "select_source_images",
        lambda *, target: calls.append(("select", target)),
    )
    monkeypatch.setattr(
        etl_module,
        "create_pill_crops",
        lambda *, overwrite: calls.append(("crop", overwrite)),
    )
    monkeypatch.setattr(
        etl_module,
        "assign_data_splits",
        lambda package: calls.append(("split", package)),
    )
    monkeypatch.setattr(
        etl_module,
        "summarize_crops",
        lambda *, minimum: calls.append(("summarize", minimum)),
    )
    monkeypatch.setattr(
        etl_module,
        "export_training_package",
        lambda *, overwrite: calls.append(("export", overwrite)),
    )

    etl_module.build_classifier_dataset(
        detector_package=detector_package,
        target_samples_per_class=120,
        skip_png_validation=True,
        overwrite=True,
    )

    assert calls == [
        ("validate", True),
        ("select", 120),
        ("crop", True),
        ("split", detector_package),
        ("summarize", 120),
        ("export", True),
    ]


def test_classifier_etl_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError, match="target_samples_per_class"):
        etl_module.build_classifier_dataset(target_samples_per_class=0)
