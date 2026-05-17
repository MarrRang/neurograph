"""Path helpers that keep NeuroGraph project-local."""

from __future__ import annotations

from pathlib import Path

from neurograph import config


def project_root(path: Path | None = None) -> Path:
    """Return the resolved project root for a CLI invocation."""

    return (path or Path.cwd()).resolve()


def neurograph_dir(root: Path) -> Path:
    return root / config.APP_DIR_NAME


def cache_dir(root: Path) -> Path:
    return neurograph_dir(root) / config.CACHE_DIR_NAME


def context_dir(root: Path) -> Path:
    return neurograph_dir(root) / config.CONTEXT_DIR_NAME


def manifest_path(root: Path) -> Path:
    return neurograph_dir(root) / config.MANIFEST_NAME


def db_path(root: Path) -> Path:
    return neurograph_dir(root) / config.DB_NAME


def ignore_path(root: Path) -> Path:
    return root / config.IGNORE_FILE_NAME


def as_project_path(root: Path, path: Path) -> str:
    """Return a stable POSIX path relative to the project root."""

    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_recorded_path(root: Path, recorded_path: str) -> Path:
    """Resolve a manifest path, rejecting absolute or escaping paths."""

    candidate = Path(recorded_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Refusing unsafe manifest path: {recorded_path}")
    resolved = (root / candidate).resolve()
    resolved.relative_to(root.resolve())
    return resolved
