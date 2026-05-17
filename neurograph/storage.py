"""DuckDB-backed local storage for NeuroGraph."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import duckdb

from neurograph.graph.schema import Chunk, Edge, FileRecord
from neurograph.utils.hashing import sha256_text
from neurograph.utils.paths import db_path


JsonValue = Mapping[str, Any] | list[Any] | str | int | float | bool | None


def connect(root: Path) -> duckdb.DuckDBPyConnection:
    """Connect to the project-local DuckDB file."""

    return duckdb.connect(str(db_path(root)))


def init_db(root: Path) -> None:
    """Create the NeuroGraph v0.1 schema if needed.

    DuckDB table creation is idempotent, so calling this from repeated
    `ng init` runs does not duplicate data.
    """

    db_path(root).parent.mkdir(parents=True, exist_ok=True)
    with connect(root) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                uri TEXT,
                path TEXT,
                kind TEXT,
                title TEXT,
                content_hash TEXT,
                indexed_at TIMESTAMP,
                metadata JSON
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                kind TEXT,
                label TEXT,
                canonical_name TEXT,
                artifact_id TEXT,
                path TEXT,
                start_line INTEGER,
                end_line INTEGER,
                page INTEGER,
                bbox JSON,
                confidence TEXT,
                metadata JSON
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                src TEXT,
                dst TEXT,
                relation TEXT,
                confidence TEXT,
                score DOUBLE,
                extractor TEXT,
                evidence_id TEXT,
                metadata JSON
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                artifact_id TEXT,
                kind TEXT,
                text TEXT,
                summary TEXT,
                page INTEGER,
                start_line INTEGER,
                end_line INTEGER,
                token_count INTEGER,
                metadata JSON
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                artifact_id TEXT,
                source_uri TEXT,
                quote TEXT,
                start_line INTEGER,
                end_line INTEGER,
                page INTEGER,
                bbox JSON,
                extractor TEXT,
                confidence TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS context_packs (
                id TEXT PRIMARY KEY,
                task TEXT,
                mode TEXT,
                token_budget INTEGER,
                payload_json JSON,
                created_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS manifests (
                key TEXT PRIMARY KEY,
                value JSON,
                updated_at TIMESTAMP
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_path ON artifacts(path)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_artifact_id ON chunks(artifact_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_nodes_artifact_id ON nodes(artifact_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_evidence_artifact_id ON evidence(artifact_id)")
    upsert_manifest(root, "schema_version", {"version": 1})


def initialize(root: Path) -> None:
    """Backward-compatible alias used by the CLI lifecycle."""

    init_db(root)


def upsert_artifact(
    root: Path,
    *,
    id: str,
    uri: str | None = None,
    path: str | None = None,
    kind: str | None = None,
    title: str | None = None,
    content_hash: str | None = None,
    indexed_at: datetime | None = None,
    metadata: JsonValue = None,
) -> None:
    init_db(root)
    timestamp = indexed_at or _now()
    with connect(root) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO artifacts
            (id, uri, path, kind, title, content_hash, indexed_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON))
            """,
            [id, uri, path, kind, title, content_hash, timestamp, _json(metadata)],
        )


def upsert_node(
    root: Path,
    *,
    id: str,
    kind: str | None = None,
    label: str | None = None,
    canonical_name: str | None = None,
    artifact_id: str | None = None,
    path: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    page: int | None = None,
    bbox: JsonValue = None,
    confidence: str | None = None,
    metadata: JsonValue = None,
) -> None:
    init_db(root)
    with connect(root) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO nodes
            (id, kind, label, canonical_name, artifact_id, path, start_line, end_line, page, bbox, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?, CAST(? AS JSON))
            """,
            [
                id,
                kind,
                label,
                canonical_name,
                artifact_id,
                path,
                start_line,
                end_line,
                page,
                _json(bbox),
                confidence,
                _json(metadata),
            ],
        )


def upsert_edge(
    root: Path,
    *,
    id: str,
    src: str | None = None,
    dst: str | None = None,
    relation: str | None = None,
    confidence: str | None = None,
    score: float | None = None,
    extractor: str | None = None,
    evidence_id: str | None = None,
    metadata: JsonValue = None,
) -> None:
    init_db(root)
    with connect(root) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO edges
            (id, src, dst, relation, confidence, score, extractor, evidence_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON))
            """,
            [id, src, dst, relation, confidence, score, extractor, evidence_id, _json(metadata)],
        )


def upsert_chunk(
    root: Path,
    *,
    id: str,
    artifact_id: str | None = None,
    kind: str | None = None,
    text: str | None = None,
    summary: str | None = None,
    page: int | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    token_count: int | None = None,
    metadata: JsonValue = None,
) -> None:
    init_db(root)
    chunk_text = text or ""
    tokens = token_count if token_count is not None else _token_count(chunk_text)
    with connect(root) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO chunks
            (id, artifact_id, kind, text, summary, page, start_line, end_line, token_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON))
            """,
            [id, artifact_id, kind, chunk_text, summary, page, start_line, end_line, tokens, _json(metadata)],
        )


def upsert_evidence(
    root: Path,
    *,
    id: str,
    artifact_id: str | None = None,
    source_uri: str | None = None,
    quote: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    page: int | None = None,
    bbox: JsonValue = None,
    extractor: str | None = None,
    confidence: str | None = None,
) -> None:
    init_db(root)
    with connect(root) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO evidence
            (id, artifact_id, source_uri, quote, start_line, end_line, page, bbox, extractor, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?, ?)
            """,
            [
                id,
                artifact_id,
                source_uri,
                quote,
                start_line,
                end_line,
                page,
                _json(bbox),
                extractor,
                confidence,
            ],
        )


def upsert_manifest(root: Path, key: str, value: JsonValue) -> None:
    db_path(root).parent.mkdir(parents=True, exist_ok=True)
    with connect(root) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS manifests (
                key TEXT PRIMARY KEY,
                value JSON,
                updated_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            INSERT OR REPLACE INTO manifests (key, value, updated_at)
            VALUES (?, CAST(? AS JSON), ?)
            """,
            [key, _json(value), _now()],
        )


def save_context_pack(
    root: Path,
    *,
    id: str,
    task: str,
    mode: str,
    token_budget: int,
    payload_json: JsonValue,
    created_at: datetime | None = None,
) -> None:
    init_db(root)
    with connect(root) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO context_packs
            (id, task, mode, token_budget, payload_json, created_at)
            VALUES (?, ?, ?, ?, CAST(? AS JSON), ?)
            """,
            [id, task, mode, token_budget, _json(payload_json), created_at or _now()],
        )


def get_counts(root: Path) -> dict[str, Any]:
    if not db_path(root).exists():
        return _empty_counts()
    init_db(root)
    with connect(root) as con:
        artifact_total = con.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        node_total = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_total = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        chunk_total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        evidence_total = con.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
        context_pack_total = con.execute("SELECT COUNT(*) FROM context_packs").fetchone()[0]
        manifest_total = con.execute("SELECT COUNT(*) FROM manifests").fetchone()[0]
        by_kind_rows = con.execute(
            "SELECT kind, COUNT(*) FROM artifacts GROUP BY kind ORDER BY kind"
        ).fetchall()
    return {
        "artifacts": artifact_total,
        "nodes": node_total,
        "edges": edge_total,
        "chunks": chunk_total,
        "evidence": evidence_total,
        "context_packs": context_pack_total,
        "manifests": manifest_total,
        "by_kind": {kind: count for kind, count in by_kind_rows if kind is not None},
    }


def get_changed_files(root: Path, current_hashes: Mapping[str, str] | None = None) -> dict[str, list[str]]:
    """Compare current path hashes against stored artifact hashes."""

    indexed = indexed_hashes(root)
    if current_hashes is None:
        return {"new": [], "modified": [], "deleted": []}

    new = sorted(path for path in current_hashes if path not in indexed)
    modified = sorted(
        path for path, digest in current_hashes.items() if indexed.get(path) not in (None, digest)
    )
    deleted = sorted(path for path in indexed if path not in current_hashes)
    return {"new": new, "modified": modified, "deleted": deleted}


def clear_artifact_index(root: Path, path: str) -> int:
    """Remove one artifact and all derived rows for a project-relative path."""

    if not db_path(root).exists():
        return 0
    init_db(root)
    with connect(root) as con:
        artifact_rows = con.execute("SELECT id FROM artifacts WHERE path = ?", [path]).fetchall()
        artifact_ids = [row[0] for row in artifact_rows]
        if not artifact_ids:
            return 0

        placeholders = ", ".join("?" for _ in artifact_ids)
        node_ids = [
            row[0]
            for row in con.execute(
                f"SELECT id FROM nodes WHERE artifact_id IN ({placeholders})",
                artifact_ids,
            ).fetchall()
        ]
        chunk_ids = [
            row[0]
            for row in con.execute(
                f"SELECT id FROM chunks WHERE artifact_id IN ({placeholders})",
                artifact_ids,
            ).fetchall()
        ]
        evidence_ids = [
            row[0]
            for row in con.execute(
                f"SELECT id FROM evidence WHERE artifact_id IN ({placeholders})",
                artifact_ids,
            ).fetchall()
        ]
        related_ids = artifact_ids + node_ids + chunk_ids + evidence_ids

        if related_ids:
            related_placeholders = ", ".join("?" for _ in related_ids)
            con.execute(
                f"DELETE FROM edges WHERE src IN ({related_placeholders}) OR dst IN ({related_placeholders})",
                related_ids + related_ids,
            )
        if evidence_ids:
            evidence_placeholders = ", ".join("?" for _ in evidence_ids)
            con.execute(f"DELETE FROM edges WHERE evidence_id IN ({evidence_placeholders})", evidence_ids)

        con.execute(f"DELETE FROM nodes WHERE artifact_id IN ({placeholders})", artifact_ids)
        con.execute(f"DELETE FROM chunks WHERE artifact_id IN ({placeholders})", artifact_ids)
        con.execute(f"DELETE FROM evidence WHERE artifact_id IN ({placeholders})", artifact_ids)
        con.execute(f"DELETE FROM artifacts WHERE id IN ({placeholders})", artifact_ids)
        return len(artifact_ids)


def search_text(root: Path, query: str, limit: int = 10, *, allow_fts: bool = True) -> list[dict[str, Any]]:
    """Search chunks with DuckDB FTS when available, otherwise LIKE fallback."""

    if not db_path(root).exists() or not query.strip():
        return []
    with connect(root) as con:
        if allow_fts:
            try:
                hits = _try_fts_search(con, query, limit)
                if hits:
                    return hits
            except Exception:
                pass
        return _like_search(con, query, limit)


def replace_file_index(
    root: Path,
    record: FileRecord,
    chunks: Iterable[Chunk],
    edges: Iterable[Edge] = (),
) -> None:
    """Replace all derived index rows for a parsed file.

    This adapter keeps the first scaffold's indexer working while storing data
    in the v0.1 artifact/chunk/evidence graph schema.
    """

    init_db(root)
    artifact_id = artifact_id_for_path(record.path)
    clear_artifact_index(root, record.path)
    upsert_artifact(
        root,
        id=artifact_id,
        uri=record.path,
        path=record.path,
        kind=record.kind,
        title=record.title,
        content_hash=record.sha256,
        metadata={
            "size_bytes": record.size_bytes,
            "mtime": record.mtime,
            "summary": record.summary,
        },
    )

    chunk_by_id: dict[str, Chunk] = {}
    for chunk in chunks:
        chunk_by_id[chunk.id] = chunk
        evidence_id = evidence_id_for_chunk(chunk.id)
        document_node = chunk.metadata.get("document_node") if isinstance(chunk.metadata, dict) else None
        extractor = str(chunk.metadata.get("extractor") or "neurograph.v0.local") if isinstance(chunk.metadata, dict) else "neurograph.v0.local"
        chunk_metadata = {
            "file_path": chunk.file_path,
            "evidence": chunk.evidence,
            "symbol": chunk.symbol,
            "heading": chunk.heading,
            **(chunk.metadata if isinstance(chunk.metadata, dict) else {}),
        }
        confidence = "medium"
        if isinstance(document_node, dict):
            confidence = str(document_node.get("confidence") or confidence)
        upsert_chunk(
            root,
            id=chunk.id,
            artifact_id=artifact_id,
            kind=chunk.kind,
            text=chunk.text,
            page=chunk.page,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            metadata=chunk_metadata,
        )
        upsert_evidence(
            root,
            id=evidence_id,
            artifact_id=artifact_id,
            source_uri=chunk.file_path,
            quote=chunk.text,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            page=chunk.page,
            extractor=extractor,
            confidence=confidence,
        )
        if isinstance(document_node, dict):
            upsert_node(
                root,
                id=chunk.id,
                kind=str(document_node.get("kind") or "Document"),
                label=str(document_node.get("label") or chunk.heading or chunk.symbol or chunk.file_path),
                canonical_name=str(document_node.get("canonical_name") or document_node.get("label") or chunk.heading or chunk.symbol or chunk.file_path),
                artifact_id=artifact_id,
                path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                page=chunk.page,
                confidence=confidence,
                metadata=document_node.get("metadata") if isinstance(document_node.get("metadata"), dict) else {},
            )
        elif chunk.symbol or chunk.heading:
            label = chunk.symbol or chunk.heading
            upsert_node(
                root,
                id=node_id_for_chunk(chunk.id),
                kind="symbol" if chunk.symbol else "section",
                label=label,
                canonical_name=label,
                artifact_id=artifact_id,
                path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                page=chunk.page,
                confidence="medium",
                metadata={"chunk_id": chunk.id},
            )

    for edge in edges:
        target_chunk = chunk_by_id.get(edge.target)
        target_metadata = target_chunk.metadata if target_chunk and isinstance(target_chunk.metadata, dict) else {}
        target_node = target_metadata.get("document_node")
        target_confidence = "medium"
        if isinstance(target_node, dict):
            target_confidence = str(target_node.get("confidence") or target_confidence)
        target_extractor = str(target_metadata.get("extractor") or "neurograph.v0.local")
        upsert_edge(
            root,
            id=edge_id(edge.source, edge.target, edge.kind),
            src=artifact_id_for_path(edge.source) if edge.source == record.path else edge.source,
            dst=edge.target,
            relation=edge.kind,
            confidence=target_confidence,
            score=1.0,
            extractor=target_extractor,
            evidence_id=evidence_id_for_chunk(target_chunk.id) if target_chunk else None,
            metadata={"evidence": edge.evidence},
        )


def remove_missing_files(root: Path, existing_paths: set[str]) -> int:
    if not db_path(root).exists():
        return 0
    init_db(root)
    with connect(root) as con:
        rows = con.execute("SELECT path FROM artifacts").fetchall()
    missing = [row[0] for row in rows if row[0] is not None and row[0] not in existing_paths]
    removed = 0
    for path in missing:
        removed += clear_artifact_index(root, path)
    return removed


def file_counts(root: Path) -> dict[str, int]:
    counts = get_counts(root)
    by_kind = Counter(counts.get("by_kind", {}))
    by_kind["total"] = counts.get("artifacts", 0)
    return dict(by_kind)


def indexed_hashes(root: Path) -> dict[str, str]:
    if not db_path(root).exists():
        return {}
    with connect(root) as con:
        rows = con.execute(
            "SELECT path, content_hash FROM artifacts WHERE path IS NOT NULL AND content_hash IS NOT NULL"
        ).fetchall()
    return {path: digest for path, digest in rows}


def all_chunks(root: Path) -> list[Chunk]:
    if not db_path(root).exists():
        return []
    with connect(root) as con:
        rows = con.execute(
            """
            SELECT c.id, a.path, c.kind, c.text, c.page, c.start_line, c.end_line, c.metadata
            FROM chunks c
            LEFT JOIN artifacts a ON a.id = c.artifact_id
            """
        ).fetchall()
    return [_row_to_chunk(row) for row in rows]


def fetch_chunks(root: Path, ids: Iterable[str]) -> list[Chunk]:
    id_list = list(ids)
    if not id_list or not db_path(root).exists():
        return []
    placeholders = ", ".join("?" for _ in id_list)
    with connect(root) as con:
        rows = con.execute(
            f"""
            SELECT c.id, a.path, c.kind, c.text, c.page, c.start_line, c.end_line, c.metadata
            FROM chunks c
            LEFT JOIN artifacts a ON a.id = c.artifact_id
            WHERE c.id IN ({placeholders})
            """,
            id_list,
        ).fetchall()
    return [_row_to_chunk(row) for row in rows]


def artifact_id_for_path(path: str) -> str:
    return f"artifact:{sha256_text(path)[:24]}"


def evidence_id_for_chunk(chunk_id: str) -> str:
    return f"evidence:{sha256_text(chunk_id)[:24]}"


def node_id_for_chunk(chunk_id: str) -> str:
    return f"node:{sha256_text(chunk_id)[:24]}"


def edge_id(src: str, dst: str, relation: str) -> str:
    value = "\n".join([src, dst, relation])
    return f"edge:{sha256_text(value)[:24]}"


def _try_fts_search(con: duckdb.DuckDBPyConnection, query: str, limit: int) -> list[dict[str, Any]]:
    con.execute("LOAD fts")
    con.execute("PRAGMA create_fts_index('chunks', 'id', 'text', overwrite=1)")
    rows = con.execute(
        """
        SELECT
            c.id,
            c.artifact_id,
            a.path,
            c.kind,
            c.text,
            c.summary,
            c.page,
            c.start_line,
            c.end_line,
            c.token_count,
            c.metadata,
            fts_main_chunks.match_bm25(c.id, ?) AS score
        FROM chunks c
        LEFT JOIN artifacts a ON a.id = c.artifact_id
        WHERE score IS NOT NULL
        ORDER BY score DESC, c.id
        LIMIT ?
        """,
        [query, limit],
    ).fetchall()
    return [_search_row_to_dict(row, "fts") for row in rows]


def _like_search(con: duckdb.DuckDBPyConnection, query: str, limit: int) -> list[dict[str, Any]]:
    normalized = query.strip().lower()
    tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9_.$/-]+", normalized)]
    patterns = [f"%{normalized}%"] if normalized else []
    patterns.extend(f"%{token}%" for token in tokens if token != normalized)
    if not patterns:
        return []

    clauses = " OR ".join("lower(c.text) LIKE ?" for _ in patterns)
    rows = con.execute(
        f"""
        SELECT
            c.id,
            c.artifact_id,
            a.path,
            c.kind,
            c.text,
            c.summary,
            c.page,
            c.start_line,
            c.end_line,
            c.token_count,
            c.metadata,
            1.0 AS score
        FROM chunks c
        LEFT JOIN artifacts a ON a.id = c.artifact_id
        WHERE {clauses}
        ORDER BY c.id
        LIMIT ?
        """,
        [*patterns, limit],
    ).fetchall()
    return [_search_row_to_dict(row, "like") for row in rows]


def _search_row_to_dict(row: tuple[Any, ...], method: str) -> dict[str, Any]:
    return {
        "id": row[0],
        "artifact_id": row[1],
        "path": row[2],
        "kind": row[3],
        "text": row[4],
        "summary": row[5],
        "page": row[6],
        "start_line": row[7],
        "end_line": row[8],
        "token_count": row[9],
        "metadata": _json_load(row[10]),
        "score": float(row[11]) if row[11] is not None else 0.0,
        "search_method": method,
    }


def _row_to_chunk(row: tuple[Any, ...]) -> Chunk:
    metadata = _json_load(row[7])
    file_path = metadata.get("file_path") or row[1] or ""
    evidence = metadata.get("evidence") or _evidence_for(file_path, row[5], row[6], row[4])
    return Chunk(
        id=row[0],
        file_path=file_path,
        kind=row[2],
        text=row[3] or "",
        evidence=evidence,
        symbol=metadata.get("symbol"),
        heading=metadata.get("heading"),
        page=row[4],
        start_line=row[5],
        end_line=row[6],
    )


def _evidence_for(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    page: int | None = None,
) -> str:
    if page is not None:
        return f"{path}:page {page}"
    if start_line is not None and end_line is not None:
        if start_line == end_line:
            return f"{path}:{start_line}"
        return f"{path}:{start_line}-{end_line}"
    return path


def _json(value: JsonValue) -> str:
    if value is None:
        value = {}
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_load(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _empty_counts() -> dict[str, Any]:
    return {
        "artifacts": 0,
        "nodes": 0,
        "edges": 0,
        "chunks": 0,
        "evidence": 0,
        "context_packs": 0,
        "manifests": 0,
        "by_kind": {},
    }
