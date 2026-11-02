"""프로젝트 문서의 저장소 내부 링크를 검사한다."""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^]]*]\(([^)]+)\)")


def test_markdown_relative_links_resolve() -> None:
    """README, docs와 reports의 상대 링크가 실제 경로를 가리키는지 확인한다."""

    markdown_paths = [
        PROJECT_ROOT / "README.md",
        *sorted((PROJECT_ROOT / "docs").rglob("*.md")),
        *sorted((PROJECT_ROOT / "reports").rglob("*.md")),
    ]
    broken_links: list[str] = []

    for markdown_path in markdown_paths:
        content = markdown_path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK_PATTERN.findall(content):
            target = raw_target.strip().split("#", maxsplit=1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved_path = (markdown_path.parent / target).resolve()
            if resolved_path.exists():
                continue
            source = markdown_path.relative_to(PROJECT_ROOT).as_posix()
            broken_links.append(f"{source}: {raw_target}")

    assert not broken_links, "깨진 문서 링크:\n" + "\n".join(broken_links)
