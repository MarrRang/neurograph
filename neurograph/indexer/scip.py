"""Optional SCIP precise-symbol overlay importer.

v0.1 treats SCIP as a best-effort precision layer. If a JSON/SCIP-like
fixture is available, this importer records compiler-confidence symbols and
references. Binary protobuf SCIP files are skipped safely until a lightweight
decoder is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from neurograph import storage
from neurograph.utils.hashing import sha256_text
from neurograph.utils.paths import as_project_path


EXACT_COMPILER = "EXACT_COMPILER"
EXTRACTOR = "scip"
COMMON_SCIP_LOCATIONS = (
    "index.scip",
    ".scip/index.scip",
    ".neurograph/index.scip",
)


@dataclass(frozen=True)
class ScipImportResult:
    path: str | None
    imported_symbols: int = 0
    imported_edges: int = 0
    status: str = "missing"
    message: str | None = None


@dataclass(frozen=True)
class ScipOccurrence:
    symbol: str
    document_path: str
    start_line: int | None
    end_line: int | None
    quote: str
    relation: str
    role: str
    syntax_kind: str | None
    metadata: dict[str, Any]


def detect_scip_file(root: Path, explicit: Path | None = None) -> Path | None:
    """Return an explicit or common SCIP file if it exists."""

    root = root.resolve()
    if explicit is not None:
        path = explicit if explicit.is_absolute() else root / explicit
        return path.resolve() if path.exists() and path.is_file() else None

    for candidate in COMMON_SCIP_LOCATIONS:
        path = root / candidate
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def import_scip_overlay(root: Path, scip_path: Path | None = None) -> ScipImportResult:
    """Import a best-effort SCIP overlay without requiring SCIP support."""

    root = root.resolve()
    detected = detect_scip_file(root, scip_path)
    if detected is None:
        clear_scip_overlay(root)
        return ScipImportResult(path=str(scip_path) if scip_path else None, status="missing")

    try:
        payload = _read_json_payload(detected)
    except UnicodeDecodeError:
        clear_scip_overlay(root)
        return ScipImportResult(path=_display_path(root, detected), status="unsupported_binary", message="binary SCIP decoding is not enabled")
    except json.JSONDecodeError as exc:
        clear_scip_overlay(root)
        return ScipImportResult(path=_display_path(root, detected), status="unsupported_format", message=str(exc))
    except OSError as exc:
        return ScipImportResult(path=str(detected), status="unreadable", message=str(exc))

    clear_scip_overlay(root)
    documents = _documents(payload)
    symbol_count = 0
    edge_count = 0
    seen_symbols: set[str] = set()
    seen_edges: set[str] = set()

    for document in documents:
        document_path = _document_path(root, document)
        if not document_path:
            continue
        source_lines = _source_lines(root, document_path)

        for symbol_info in _document_symbols(document):
            symbol = _symbol_value(symbol_info)
            if not symbol:
                continue
            if symbol not in seen_symbols:
                _upsert_symbol_node(root, document_path, symbol, symbol_info, source_lines, None)
                seen_symbols.add(symbol)
                symbol_count += 1

        for occurrence in _occurrences(document, document_path, source_lines):
            if not occurrence.symbol:
                continue
            if occurrence.symbol not in seen_symbols:
                _upsert_symbol_node(root, document_path, occurrence.symbol, occurrence.metadata, source_lines, occurrence)
                seen_symbols.add(occurrence.symbol)
                symbol_count += 1
            else:
                _upsert_symbol_node(root, document_path, occurrence.symbol, occurrence.metadata, source_lines, occurrence)

            edge_id = _upsert_occurrence_edge(root, occurrence)
            if edge_id and edge_id not in seen_edges:
                seen_edges.add(edge_id)
                edge_count += 1

    return ScipImportResult(
        path=_display_path(root, detected),
        imported_symbols=symbol_count,
        imported_edges=edge_count,
        status="imported",
    )


def clear_scip_overlay(root: Path) -> int:
    """Remove rows owned by the SCIP overlay while leaving the fast graph intact."""

    storage.init_db(root)
    with storage.connect(root) as con:
        edge_count = con.execute("SELECT COUNT(*) FROM edges WHERE id LIKE 'scip:%' OR extractor = 'scip'").fetchone()[0]
        con.execute("DELETE FROM edges WHERE id LIKE 'scip:%' OR extractor = 'scip'")
        con.execute("DELETE FROM nodes WHERE id LIKE 'scip:%'")
        con.execute("DELETE FROM evidence WHERE id LIKE 'scip:%' OR extractor = 'scip'")
    return int(edge_count or 0)


def _read_json_payload(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(root: Path, path: Path) -> str:
    try:
        return as_project_path(root, path)
    except ValueError:
        return str(path)


def _documents(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    documents = payload.get("documents")
    if isinstance(documents, list):
        return tuple(item for item in documents if isinstance(item, Mapping))
    if all(key in payload for key in ("path", "occurrences")):
        return (payload,)
    return ()


def _document_path(root: Path, document: Mapping[str, Any]) -> str | None:
    for key in ("relative_path", "relativePath", "path", "uri"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_document_path(root, value)
    return None


def _normalize_document_path(root: Path, value: str) -> str:
    cleaned = value.strip().removeprefix("file://").replace("\\", "/")
    path = Path(cleaned)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _document_symbols(document: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    symbols = document.get("symbols") or document.get("symbolInformation") or document.get("symbol_information")
    if not isinstance(symbols, list):
        return ()
    return tuple(item for item in symbols if isinstance(item, Mapping))


def _occurrence_items(document: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    occurrences = document.get("occurrences")
    if not isinstance(occurrences, list):
        return ()
    return tuple(item for item in occurrences if isinstance(item, Mapping))


def _occurrences(document: Mapping[str, Any], document_path: str, source_lines: list[str]) -> tuple[ScipOccurrence, ...]:
    results: list[ScipOccurrence] = []
    for item in _occurrence_items(document):
        symbol = _symbol_value(item)
        if not symbol:
            continue
        start_line, end_line = _line_range(item.get("range") or item.get("enclosing_range") or item.get("enclosingRange"))
        roles = _roles(item)
        relation = _relation_for_roles(roles)
        role = ",".join(sorted(roles)) if roles else "reference"
        syntax_kind = _syntax_kind(item)
        quote = _quote(source_lines, start_line, end_line) or _display_name(symbol, item)
        results.append(
            ScipOccurrence(
                symbol=symbol,
                document_path=document_path,
                start_line=start_line,
                end_line=end_line,
                quote=quote,
                relation=relation,
                role=role,
                syntax_kind=syntax_kind,
                metadata={key: _jsonable(value) for key, value in item.items() if key not in {"range", "enclosing_range", "enclosingRange"}},
            )
        )
    return tuple(results)


def _symbol_value(item: Mapping[str, Any]) -> str | None:
    for key in ("symbol", "symbol_id", "symbolId", "id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _line_range(value: Any) -> tuple[int | None, int | None]:
    if not isinstance(value, list) or not value or not all(isinstance(item, int) for item in value[:4]):
        return None, None
    start = max(1, int(value[0]) + 1)
    if len(value) >= 4:
        end = max(start, int(value[2]) + 1)
    else:
        end = start
    return start, end


def _roles(item: Mapping[str, Any]) -> set[str]:
    roles: set[str] = set()
    for key in ("role", "roles", "symbol_roles", "symbolRoles"):
        value = item.get(key)
        roles.update(_role_tokens(value))
    if item.get("definition") is True or item.get("is_definition") is True or item.get("isDefinition") is True:
        roles.add("definition")
    return roles


def _role_tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, int):
        tokens = {"definition"} if value & 1 else set()
        if value & 64:
            tokens.add("definition")
        return tokens
    if isinstance(value, str):
        return {value.lower().replace(" ", "_")}
    if isinstance(value, list):
        tokens: set[str] = set()
        for item in value:
            tokens.update(_role_tokens(item))
        return tokens
    if isinstance(value, Mapping):
        return {str(key).lower().replace(" ", "_") for key, enabled in value.items() if enabled}
    return set()


def _relation_for_roles(roles: set[str]) -> str:
    if any("implement" in role for role in roles):
        return "IMPLEMENTS"
    if any("definition" in role for role in roles):
        return "DEFINES"
    return "REFERENCES"


def _syntax_kind(item: Mapping[str, Any]) -> str | None:
    for key in ("syntax_kind", "syntaxKind", "kind"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return None


def _upsert_symbol_node(
    root: Path,
    document_path: str,
    symbol: str,
    symbol_info: Mapping[str, Any],
    source_lines: list[str],
    occurrence: ScipOccurrence | None,
) -> None:
    start_line = occurrence.start_line if occurrence else None
    end_line = occurrence.end_line if occurrence else None
    if start_line is None or end_line is None:
        start_line, end_line = _line_range(symbol_info.get("range") or symbol_info.get("enclosing_range") or symbol_info.get("enclosingRange"))
    quote = occurrence.quote if occurrence else (_quote(source_lines, start_line, end_line) or _display_name(symbol, symbol_info))
    storage.upsert_node(
        root,
        id=_symbol_node_id(symbol),
        kind="Symbol",
        label=_display_name(symbol, symbol_info),
        canonical_name=symbol,
        artifact_id=storage.artifact_id_for_path(document_path),
        path=document_path,
        start_line=start_line,
        end_line=end_line,
        confidence=EXACT_COMPILER,
        metadata={
            "extractor": EXTRACTOR,
            "scip_symbol": symbol,
            "scip_kind": _syntax_kind(symbol_info),
            "quote": quote,
        },
    )


def _upsert_occurrence_edge(root: Path, occurrence: ScipOccurrence) -> str | None:
    symbol_node_id = _symbol_node_id(occurrence.symbol)
    artifact_id = storage.artifact_id_for_path(occurrence.document_path)
    source_id = _matching_code_node_id(root, occurrence) or _file_node_id(root, occurrence.document_path) or artifact_id
    evidence_id = _evidence_id(occurrence)
    edge_id = _edge_id(source_id, symbol_node_id, occurrence.relation, evidence_id)

    storage.upsert_evidence(
        root,
        id=evidence_id,
        artifact_id=artifact_id,
        source_uri=occurrence.document_path,
        quote=occurrence.quote,
        start_line=occurrence.start_line,
        end_line=occurrence.end_line,
        extractor=EXTRACTOR,
        confidence=EXACT_COMPILER,
    )
    storage.upsert_edge(
        root,
        id=edge_id,
        src=source_id,
        dst=symbol_node_id,
        relation=occurrence.relation,
        confidence=EXACT_COMPILER,
        score=1.0,
        extractor=EXTRACTOR,
        evidence_id=evidence_id,
        metadata={
            "role": occurrence.role,
            "source_path": occurrence.document_path,
            "start_line": occurrence.start_line,
            "end_line": occurrence.end_line,
            "syntax_kind": occurrence.syntax_kind,
        },
    )
    return edge_id


def _matching_code_node_id(root: Path, occurrence: ScipOccurrence) -> str | None:
    label = _display_name(occurrence.symbol, occurrence.metadata)
    if not label:
        return None
    storage.init_db(root)
    with storage.connect(root) as con:
        rows = con.execute(
            """
            SELECT id, kind, label, start_line, end_line
            FROM nodes
            WHERE path = ?
              AND kind IN ('Class', 'Function', 'Method')
            """,
            [occurrence.document_path],
        ).fetchall()

    candidates: list[tuple[int, str]] = []
    for node_id, kind, node_label, start_line, end_line in rows:
        score = 0
        if node_label == label:
            score -= 100
        elif label and str(node_label or "").lower() == label.lower():
            score -= 75
        if occurrence.start_line is not None and start_line is not None:
            score += abs(int(start_line) - occurrence.start_line)
        else:
            score += 1000
        if occurrence.end_line is not None and end_line is not None:
            score += min(abs(int(end_line) - occurrence.end_line), 20)
        if kind == _node_kind_hint(occurrence):
            score -= 10
        candidates.append((score, str(node_id)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_score, best_id = candidates[0]
    return best_id if best_score < 1000 else None


def _file_node_id(root: Path, document_path: str) -> str | None:
    storage.init_db(root)
    with storage.connect(root) as con:
        row = con.execute(
            """
            SELECT id
            FROM nodes
            WHERE path = ? AND kind = 'File'
            ORDER BY id
            LIMIT 1
            """,
            [document_path],
        ).fetchone()
    return str(row[0]) if row else None


def _node_kind_hint(occurrence: ScipOccurrence) -> str | None:
    value = (occurrence.syntax_kind or "").lower()
    if "class" in value:
        return "Class"
    if "method" in value:
        return "Method"
    if "function" in value:
        return "Function"
    return None


def _source_lines(root: Path, document_path: str) -> list[str]:
    source = Path(document_path)
    if not source.is_absolute():
        source = root / document_path
    if not source.exists() or not source.is_file():
        return []
    return source.read_text(encoding="utf-8", errors="replace").splitlines()


def _quote(lines: list[str], start_line: int | None, end_line: int | None) -> str:
    if not lines or start_line is None:
        return ""
    start = max(1, start_line)
    end = min(len(lines), max(start, end_line or start))
    return "\n".join(lines[start - 1 : end]).strip()


def _display_name(symbol: str, item: Mapping[str, Any]) -> str:
    for key in ("display_name", "displayName", "name", "descriptor"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return _clean_display_name(value)
    tail = symbol.strip().split()[-1] if symbol.strip() else symbol
    tail = tail.strip().rstrip(".")
    for separator in ("/", "#", "."):
        if separator in tail:
            tail = tail.split(separator)[-1]
    return _clean_display_name(tail) or symbol


def _clean_display_name(value: str) -> str:
    value = value.strip().rstrip(".")
    for suffix in ("().", "()", "."):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value.strip()


def _symbol_node_id(symbol: str) -> str:
    return f"scip:symbol:{sha256_text(symbol)[:24]}"


def _evidence_id(occurrence: ScipOccurrence) -> str:
    value = "\n".join(
        [
            occurrence.document_path,
            occurrence.symbol,
            str(occurrence.start_line),
            str(occurrence.end_line),
            occurrence.relation,
            occurrence.quote,
        ]
    )
    return f"scip:evidence:{sha256_text(value)[:24]}"


def _edge_id(src: str, dst: str, relation: str, evidence_id: str) -> str:
    return f"scip:edge:{sha256_text(chr(10).join([src, dst, relation, evidence_id]))[:24]}"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)
