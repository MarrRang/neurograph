"""Deterministic project scan and indexing orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path

from neurograph import config, storage
from neurograph.graph.builder import build_file_index
from neurograph.graph.linker import link_code_documents
from neurograph.indexer.scip import import_scip_overlay
from neurograph.utils.hashing import sha256_file
from neurograph.utils.ignore import IgnoreMatcher, load_ignore_matcher
from neurograph.utils.paths import as_project_path


@dataclass(frozen=True)
class IndexTarget:
    path: Path
    rel_path: str
    kind: str
    content_hash: str = ""


@dataclass(frozen=True)
class Discovery:
    targets: tuple[IndexTarget, ...]
    total_discovered: int
    skipped_ignored: int
    skipped_unsupported: int


@dataclass(frozen=True)
class IndexPlan:
    targets: tuple[IndexTarget, ...]
    unchanged: tuple[IndexTarget, ...]
    changed: tuple[IndexTarget, ...]
    deleted: tuple[str, ...]
    total_discovered: int
    skipped_ignored: int
    skipped_unsupported: int
    kind_counts: dict[str, int]


@dataclass(frozen=True)
class IndexResult:
    indexed: int
    skipped: int
    removed: int
    total_discovered: int
    skipped_ignored: int
    skipped_unsupported: int
    kind_counts: dict[str, int]
    scip_path: str | None = None
    scip_status: str = "missing"
    scip_imported_symbols: int = 0
    scip_imported_edges: int = 0
    scip_message: str | None = None
    code_doc_links: int = 0
    semantic_candidates: int = 0
    cost: str = "$0.00"


def discover_files(root: Path) -> list[Path]:
    """Return deterministic project files that should be indexed."""

    return [target.path for target in _discover(root).targets]


def classify_file(path: Path) -> str | None:
    """Classify one path into a NeuroGraph v0.1 index kind."""

    name = path.name.lower()
    suffix = path.suffix.lower()

    if name in config.OPENAPI_FILENAMES or name.endswith((".openapi.yaml", ".openapi.yml", ".openapi.json")):
        return "openapi"
    if suffix in config.SUPPORTED_MARKDOWN_EXTENSIONS:
        return "markdown"
    if suffix in config.SUPPORTED_PDF_EXTENSIONS:
        return "pdf"
    if suffix in config.SUPPORTED_SQL_EXTENSIONS:
        return "sql"
    if suffix in config.SUPPORTED_CODE_EXTENSIONS:
        return "code"
    if suffix in config.SUPPORTED_CONFIG_EXTENSIONS:
        return "config"
    if suffix in config.SUPPORTED_SB_EXTENSIONS:
        return "markdown"
    return None


def classify(path: Path) -> str | None:
    """Backward-compatible alias for older scaffold code."""

    return classify_file(path)


def should_index(path: Path, root: Path | None = None, matcher: IgnoreMatcher | None = None) -> bool:
    """Return whether a path is indexable after ignore and kind checks."""

    root = (root or Path.cwd()).resolve()
    path = path.resolve()
    if not path.is_file():
        return False
    matcher = matcher or load_ignore_matcher(root)
    if matcher.ignores(path, is_dir=False):
        return False
    return classify_file(path) is not None


def file_hash(path: Path) -> str:
    return sha256_file(path)


def build_index_plan(root: Path) -> IndexPlan:
    root = root.resolve()
    discovery = _discover(root)
    indexed = storage.indexed_hashes(root)

    changed: list[IndexTarget] = []
    unchanged: list[IndexTarget] = []
    current_paths: set[str] = set()
    kind_counts: Counter[str] = Counter()

    for target in discovery.targets:
        digest = file_hash(target.path)
        hashed_target = IndexTarget(
            path=target.path,
            rel_path=target.rel_path,
            kind=target.kind,
            content_hash=digest,
        )
        current_paths.add(target.rel_path)
        kind_counts[target.kind] += 1
        if indexed.get(target.rel_path) == digest:
            unchanged.append(hashed_target)
        else:
            changed.append(hashed_target)

    deleted = tuple(sorted(path for path in indexed if path not in current_paths))

    return IndexPlan(
        targets=tuple(sorted((*changed, *unchanged), key=lambda item: item.rel_path)),
        unchanged=tuple(sorted(unchanged, key=lambda item: item.rel_path)),
        changed=tuple(sorted(changed, key=lambda item: item.rel_path)),
        deleted=deleted,
        total_discovered=discovery.total_discovered,
        skipped_ignored=discovery.skipped_ignored,
        skipped_unsupported=discovery.skipped_unsupported,
        kind_counts=dict(sorted(kind_counts.items())),
    )


def plan(root: Path) -> list[IndexTarget]:
    """Backward-compatible wrapper returning all current index targets."""

    return list(build_index_plan(root).targets)


def changed_files(root: Path) -> dict[str, list[str]]:
    index_plan = build_index_plan(root)
    indexed = storage.indexed_hashes(root)
    return {
        "new": sorted(target.rel_path for target in index_plan.changed if target.rel_path not in indexed),
        "modified": sorted(target.rel_path for target in index_plan.changed if target.rel_path in indexed),
        "deleted": list(index_plan.deleted),
    }


def run_index(root: Path, scip_path: Path | None = None) -> IndexResult:
    index_plan = build_index_plan(root)
    existing = {target.rel_path for target in index_plan.targets}
    indexed = 0

    for target in index_plan.changed:
        record, chunks, edges = build_file_index(
            root,
            target.path,
            target.rel_path,
            target.kind,
            target.content_hash,
        )
        storage.replace_file_index(root, record, chunks, edges)
        indexed += 1

    removed = storage.remove_missing_files(root, existing)
    scip_result = import_scip_overlay(root, scip_path)
    link_result = link_code_documents(root)
    return IndexResult(
        indexed=indexed,
        skipped=len(index_plan.unchanged),
        removed=removed,
        total_discovered=index_plan.total_discovered,
        skipped_ignored=index_plan.skipped_ignored,
        skipped_unsupported=index_plan.skipped_unsupported,
        kind_counts=index_plan.kind_counts,
        scip_path=scip_result.path,
        scip_status=scip_result.status,
        scip_imported_symbols=scip_result.imported_symbols,
        scip_imported_edges=scip_result.imported_edges,
        scip_message=scip_result.message,
        code_doc_links=link_result.strong_links,
        semantic_candidates=link_result.weak_candidates,
    )


def _discover(root: Path) -> Discovery:
    root = root.resolve()
    matcher = load_ignore_matcher(root)
    targets: list[IndexTarget] = []
    total_discovered = 0
    skipped_ignored = 0
    skipped_unsupported = 0

    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()

        kept_dirs: list[str] = []
        for dirname in dirnames:
            directory = current_path / dirname
            if matcher.ignores(directory, is_dir=True):
                skipped_ignored += 1
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            path = current_path / filename
            total_discovered += 1
            if matcher.ignores(path, is_dir=False):
                skipped_ignored += 1
                continue
            kind = classify_file(path)
            if kind is None:
                skipped_unsupported += 1
                continue
            targets.append(IndexTarget(path=path.resolve(), rel_path=as_project_path(root, path), kind=kind))

    return Discovery(
        targets=tuple(sorted(targets, key=lambda target: target.rel_path)),
        total_discovered=total_discovered,
        skipped_ignored=skipped_ignored,
        skipped_unsupported=skipped_unsupported,
    )
