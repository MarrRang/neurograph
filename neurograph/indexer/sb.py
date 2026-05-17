"""Plain text SB document parsing for v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SbSection:
    heading: str | None
    start_line: int
    end_line: int
    text: str


def parse_sb(path: Path) -> tuple[str | None, list[SbSection]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not text.strip():
        return None, []

    sections: list[SbSection] = []
    current_heading: str | None = None
    current_start = 1
    current_lines: list[str] = []
    title: str | None = None

    def flush(end_line: int) -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(SbSection(current_heading, current_start, end_line, body))

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("::") and stripped.endswith("::") and len(stripped) > 4:
            flush(line_number - 1)
            current_heading = stripped.strip(":").strip()
            title = title or current_heading
            current_start = line_number
            current_lines = [line]
        else:
            if not current_lines:
                current_start = line_number
            current_lines.append(line)
    flush(len(lines))
    return title, sections
