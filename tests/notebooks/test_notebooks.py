"""주 실행 Notebook의 JSON 구조와 Python 문법 테스트."""

import ast
import json
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_PATHS = sorted(NOTEBOOK_DIR.rglob("*.ipynb"))


@pytest.mark.parametrize(
    "notebook_path",
    NOTEBOOK_PATHS,
    ids=lambda path: path.relative_to(NOTEBOOK_DIR).as_posix(),
)
def test_notebook_code_cells_have_valid_python_syntax(notebook_path: Path) -> None:
    """Notebook을 JSON으로 읽고 각 Python code cell을 독립적으로 파싱한다."""

    notebook: dict[str, Any] = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        ast.parse(source, filename=f"{notebook_path.name}#cell-{cell_index}")


@pytest.mark.parametrize(
    "notebook_path",
    NOTEBOOK_PATHS,
    ids=lambda path: path.relative_to(NOTEBOOK_DIR).as_posix(),
)
def test_notebook_outputs_are_cleared(notebook_path: Path) -> None:
    """Git에 저장한 Notebook에는 실행 결과와 실행 번호를 남기지 않는다."""

    notebook: dict[str, Any] = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None, (
            f"{notebook_path.name}#{cell_index} execution_count가 남아 있습니다."
        )
        assert not cell.get("outputs"), (
            f"{notebook_path.name}#{cell_index} 출력이 남아 있습니다."
        )
