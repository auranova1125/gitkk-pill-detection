"""JSON과 CSV artifact 저장에 사용하는 공통 함수."""

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """UTF-8 JSON 파일을 읽어 Python 값으로 반환한다."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(
    path: Path,
    value: Any,
    *,
    sort_keys: bool = False,
) -> None:
    """UTF-8 JSON을 들여쓰기와 마지막 개행을 포함해 저장한다."""

    Path(path).write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=sort_keys,
        )
        + "\n",
        encoding="utf-8",
    )


def append_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    """CSV에 행을 추가하고 비어 있는 파일에는 header를 먼저 기록한다."""

    path = Path(path)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def prepare_training_run_directory(
    *,
    output_dir: Path,
    run_name: str,
    resume_from: Path | None,
    resolved_config: Mapping[str, Any],
    metadata_files: Mapping[str, Any],
) -> Path:
    """신규 학습의 기존 artifact 덮어쓰기를 막고 실행 metadata를 저장한다."""

    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume_from is None:
        artifact_names = ("last.pt", "best.pt", "metrics.csv")
        existing_artifacts = [
            run_dir / name for name in artifact_names if (run_dir / name).exists()
        ]
        if existing_artifacts:
            names = ", ".join(path.name for path in existing_artifacts)
            raise FileExistsError(
                f"run directory already contains training artifacts: {names}"
            )

    write_json(
        run_dir / "resolved_config.json",
        resolved_config,
        sort_keys=True,
    )
    for file_name, value in metadata_files.items():
        if Path(file_name).name != file_name or not file_name.endswith(".json"):
            raise ValueError("metadata file names must be plain JSON file names")
        write_json(run_dir / file_name, value, sort_keys=True)
    return run_dir
