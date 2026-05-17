"""Typed records used by indexing, storage, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FileRecord:
    path: str
    kind: str
    sha256: str
    size_bytes: int
    mtime: float
    title: str | None = None
    summary: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    file_path: str
    kind: str
    text: str
    evidence: str
    symbol: str | None = None
    heading: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str
    evidence: str


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    reason: str
