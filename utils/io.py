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
