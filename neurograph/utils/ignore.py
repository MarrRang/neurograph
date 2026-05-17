"""Small deterministic ignore matcher for v0.1 indexing."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from neurograph import config
from neurograph.utils.paths import as_project_path, ignore_path


@dataclass(frozen=True)
class IgnoreMatcher:
    root: Path
    patterns: tuple[str, ...]
    ignored_dir_names: frozenset[str]

    def ignores(self, path: Path, is_dir: bool | None = None) -> bool:
        try:
            rel = as_project_path(self.root, path)
        except ValueError:
            return True

        if any(part in self.ignored_dir_names for part in Path(rel).parts):
            return True

        ignored = False
        for pattern in self.patterns:
            normalized = pattern.strip()
            if not normalized or normalized.startswith("#"):
                continue
            negated = normalized.startswith("!")
            if negated:
                normalized = normalized[1:].strip()
            if not normalized:
                continue
            if _matches_pattern(rel, normalized, bool(is_dir)):
                ignored = not negated
        return ignored


def load_ignore_matcher(root: Path) -> IgnoreMatcher:
    patterns = list(config.DEFAULT_IGNORE_PATTERNS)
    for path in (root / ".gitignore", ignore_path(root)):
        if path.exists():
            patterns.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
    return IgnoreMatcher(
        root=root.resolve(),
        patterns=tuple(patterns),
        ignored_dir_names=frozenset(config.DEFAULT_IGNORED_DIRS),
    )


def default_ignore_text() -> str:
    lines = [
        "# NeuroGraph ignore file",
        "# Uses simple gitignore-like path and glob patterns.",
        *config.DEFAULT_IGNORE_PATTERNS,
    ]
    return "\n".join(lines) + "\n"


def _matches_pattern(rel: str, pattern: str, is_dir: bool) -> bool:
    rel_path = Path(rel)
    rel_parts = rel_path.parts
    basename = rel_path.name
    root_relative = pattern.startswith("/")
    normalized = pattern.lstrip("/")
    directory_pattern = normalized.endswith("/")
    normalized = normalized.rstrip("/")

    if not normalized:
        return False

    if directory_pattern:
        if "/" not in normalized:
            return normalized in rel_parts
        return rel == normalized or rel.startswith(f"{normalized}/")

    if root_relative:
        return fnmatch(rel, normalized)

    if "/" in normalized:
        return fnmatch(rel, normalized) or rel.startswith(f"{normalized}/")

    return fnmatch(basename, normalized) or fnmatch(rel, normalized) or (is_dir and normalized in rel_parts)
