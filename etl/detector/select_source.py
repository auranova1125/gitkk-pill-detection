"""SELECT 원천 annotation adapter 플레이스홀더."""

from pathlib import Path
from typing import Never


def load_select_annotations(source_dir: Path) -> Never:
    """SELECT 원천 형식이 확정된 뒤 구현할 adapter."""

    raise NotImplementedError(
        "SELECT source adapter is pending: provide the validated SELECT annotation "
        f"layout before reading {Path(source_dir)}"
    )
