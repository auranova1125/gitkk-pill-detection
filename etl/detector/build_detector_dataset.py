"""탐지 학습 패키지 전체 ETL 진입점 플레이스홀더."""

from pathlib import Path


def build_detector_dataset(
    *,
    select_source_dir: Path,
    codeit_source_dir: Path,
    output_dir: Path,
    existing_split_manifest: Path | None = None,
) -> None:
    """SELECT 입력과 최초 split 규칙이 확정된 뒤 전체 ETL을 연결한다."""

    del select_source_dir, codeit_source_dir, output_dir, existing_split_manifest
    raise NotImplementedError(
        "detector ETL orchestration is pending the validated SELECT source layout "
        "and the original group-split rule; the Codeit source reader and final "
        "package validator are available independently"
    )
