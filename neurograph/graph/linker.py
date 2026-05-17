"""Conservative document-to-code graph linker for v0.1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from neurograph import storage
from neurograph.utils.hashing import sha256_text


EXTRACTOR = "code_document_linker"
DOC_EXACT = "DOC_EXACT"
DOC_INFERRED = "DOC_INFERRED"
AMBIGUOUS = "AMBIGUOUS"
EXACT_STATIC = "EXACT_STATIC"

DOC_KINDS = {"markdown", "pdf", "sb"}
CODE_KINDS = {"code", "openapi", "sql", "config"}
ENDPOINT_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
GENERIC_TERMS = {
    "api",
    "app",
    "data",
    "delete",
    "get",
    "handler",
    "item",
    "list",
    "login",
    "order",
    "page",
    "payment",
    "post",
    "put",
    "request",
    "response",
    "save",
    "screen",
    "service",
    "status",
    "submit",
    "update",
    "user",
    "users",
    "view",
}
VALIDATION_TOKENS = {
    "assert",
    "cannot",
    "empty",
    "error",
    "invalid",
    "must",
    "raise",
    "required",
    "throw",
    "validate",
    "validation",
}


@dataclass(frozen=True)
class LinkResult:
    strong_links: int = 0
    weak_candidates: int = 0


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: str
    label: str
    canonical_name: str
    artifact_kind: str | None
    artifact_path: str | None
    path: str | None
    start_line: int | None
    end_line: int | None
    page: int | None
    confidence: str | None
    metadata: dict[str, Any]
    chunk_text: str


@dataclass(frozen=True)
class Link:
    source: GraphNode
    target: GraphNode
    relation: str
    confidence: str
    score: float
    reason: str
    default_context_pack: bool


def link_code_documents(root: Path) -> LinkResult:
    """Create conservative links between indexed document facts and code nodes."""

    storage.init_db(root)
    _clear_links(root)
    nodes = _load_nodes(root)
    by_id = {node.id: node for node in nodes}
    route_to_handler = _load_route_to_handler(root)

    doc_nodes = [node for node in nodes if _is_doc_node(node)]
    endpoint_targets = [node for node in nodes if node.kind == "Endpoint" and node.artifact_kind in CODE_KINDS]
    code_targets = [node for node in nodes if node.artifact_kind in CODE_KINDS]

    links: dict[str, Link] = {}

    _link_exact_endpoints(doc_nodes, endpoint_targets, links)
    _link_exact_fields_tables_and_config(doc_nodes, code_targets, links)
    _link_exact_code_references(doc_nodes, code_targets, links)
    _link_validation_rules(doc_nodes, code_targets, links)
    _link_openapi_endpoints_to_handlers(endpoint_targets, route_to_handler, by_id, links)
    _link_semantic_candidates(doc_nodes, code_targets, links)

    strong = 0
    weak = 0
    for link in links.values():
        _upsert_link(root, link)
        if link.relation == "SEMANTIC_CANDIDATE":
            weak += 1
        else:
            strong += 1

    return LinkResult(strong_links=strong, weak_candidates=weak)


def _clear_links(root: Path) -> None:
    with storage.connect(root) as con:
        con.execute("DELETE FROM edges WHERE extractor = ?", [EXTRACTOR])


def _load_nodes(root: Path) -> list[GraphNode]:
    with storage.connect(root) as con:
        rows = con.execute(
            """
            SELECT
                n.id,
                n.kind,
                n.label,
                n.canonical_name,
                a.kind AS artifact_kind,
                a.path AS artifact_path,
                n.path,
                n.start_line,
                n.end_line,
                n.page,
                n.confidence,
                n.metadata,
                c.text
            FROM nodes n
            LEFT JOIN artifacts a ON a.id = n.artifact_id
            LEFT JOIN chunks c ON c.id = n.id
            ORDER BY n.id
            """
        ).fetchall()
    return [
        GraphNode(
            id=str(row[0]),
            kind=str(row[1] or ""),
            label=str(row[2] or ""),
            canonical_name=str(row[3] or row[2] or ""),
            artifact_kind=row[4],
            artifact_path=row[5],
            path=row[6],
            start_line=row[7],
            end_line=row[8],
            page=row[9],
            confidence=row[10],
            metadata=_json_load(row[11]),
            chunk_text=str(row[12] or ""),
        )
        for row in rows
    ]


def _load_route_to_handler(root: Path) -> list[tuple[str, str]]:
    with storage.connect(root) as con:
        return [(str(row[0]), str(row[1])) for row in con.execute("SELECT src, dst FROM edges WHERE relation = 'ROUTE_TO_HANDLER'").fetchall()]


def _link_exact_endpoints(doc_nodes: list[GraphNode], endpoint_targets: list[GraphNode], links: dict[str, Link]) -> None:
    endpoints_by_key: dict[tuple[str | None, str], list[GraphNode]] = {}
    endpoints_by_path: dict[str, list[GraphNode]] = {}
    for endpoint in endpoint_targets:
        method, path = _endpoint_parts(endpoint)
        if not path:
            continue
        endpoints_by_path.setdefault(path, []).append(endpoint)
        endpoints_by_key.setdefault((method, path), []).append(endpoint)

    for doc in doc_nodes:
        if doc.kind != "APIReference":
            continue
        method, path = _endpoint_parts(doc)
        if not path:
            continue
        candidates = endpoints_by_key.get((method, path), []) if method else endpoints_by_path.get(path, [])
        for target in candidates:
            if target.id == doc.id:
                continue
            _add_link(
                links,
                Link(
                    source=doc,
                    target=target,
                    relation="SB_REFERENCES_API" if _is_sb_doc(doc) else "DOC_MENTIONS",
                    confidence=DOC_EXACT,
                    score=1.0,
                    reason="exact_endpoint",
                    default_context_pack=True,
                ),
            )


def _link_exact_fields_tables_and_config(doc_nodes: list[GraphNode], code_targets: list[GraphNode], links: dict[str, Link]) -> None:
    fields = _index_by_identifier([node for node in code_targets if node.kind in {"Column", "ConfigKey"}])
    tables = _index_by_identifier([node for node in code_targets if node.kind == "Table"])
    config_keys = _index_by_identifier([node for node in code_targets if node.kind == "ConfigKey"], include_canonical=True)
    status_values = _index_by_identifier([node for node in code_targets if node.kind in {"ConfigKey", "Column"}])

    for doc in doc_nodes:
        key = _identifier_key(doc.label)
        if not key:
            continue
        if doc.kind in {"FormField", "CodeReference"} and _doc_reference_type(doc) in {"field", "form_field", "config_key", ""}:
            for target in (*fields.get(key, ()), *config_keys.get(key, ())):
                _add_link(
                    links,
                    Link(doc, target, "DOC_MENTIONS", DOC_EXACT, 0.95, "exact_field_or_config_key", True),
                )
        if doc.kind == "DatabaseEntity":
            for target in tables.get(key, ()):
                _add_link(links, Link(doc, target, "DOC_MENTIONS", DOC_EXACT, 0.95, "exact_table", True))
        if doc.kind == "StatusValue":
            for target in status_values.get(key, ()):
                _add_link(
                    links,
                    Link(
                        doc,
                        target,
                        "SB_REFERENCES_STATUS" if _is_sb_doc(doc) else "DOC_MENTIONS",
                        DOC_EXACT,
                        0.9,
                        "exact_status_value",
                        True,
                    ),
                )


def _link_exact_code_references(doc_nodes: list[GraphNode], code_targets: list[GraphNode], links: dict[str, Link]) -> None:
    files = _index_by_path([node for node in code_targets if node.kind == "File"])
    symbols = _index_by_identifier([node for node in code_targets if node.kind in {"Class", "Function", "Method"}], include_canonical=True)

    for doc in doc_nodes:
        if doc.kind != "CodeReference":
            continue
        reference_type = _doc_reference_type(doc)
        if reference_type == "file_path":
            for target in files.get(_normalize_path(doc.label), ()):
                _add_link(links, Link(doc, target, "DOC_MENTIONS", DOC_EXACT, 1.0, "exact_file_path", True))
            continue

        label = _symbol_reference(doc.label)
        key = _identifier_key(label)
        if not key or _is_generic_single_term(label):
            continue
        for target in symbols.get(key, ()):
            relation = "DOC_MENTIONS" if doc.confidence != AMBIGUOUS else "SEMANTIC_CANDIDATE"
            _add_link(
                links,
                Link(
                    doc,
                    target,
                    relation,
                    DOC_EXACT if relation != "SEMANTIC_CANDIDATE" else AMBIGUOUS,
                    0.9 if relation != "SEMANTIC_CANDIDATE" else 0.2,
                    "exact_code_reference",
                    relation != "SEMANTIC_CANDIDATE",
                ),
            )


def _link_validation_rules(doc_nodes: list[GraphNode], code_targets: list[GraphNode], links: dict[str, Link]) -> None:
    validation_targets = [node for node in code_targets if node.kind in {"Function", "Method", "File"} and _looks_like_validation_code(node)]
    for doc in doc_nodes:
        if doc.kind != "ValidationRule":
            continue
        field = _validation_field(doc)
        if not field:
            continue
        field_key = _identifier_key(field)
        for target in validation_targets:
            if field_key and _contains_identifier(target.chunk_text, field_key):
                _add_link(
                    links,
                    Link(doc, target, "SB_DEFINES_VALIDATION", DOC_INFERRED, 0.85, "validation_field_in_validation_code", True),
                )


def _link_openapi_endpoints_to_handlers(
    endpoint_targets: list[GraphNode],
    route_to_handler: list[tuple[str, str]],
    by_id: dict[str, GraphNode],
    links: dict[str, Link],
) -> None:
    code_endpoint_handlers: dict[tuple[str | None, str], list[GraphNode]] = {}
    for endpoint_id, handler_id in route_to_handler:
        endpoint = by_id.get(endpoint_id)
        handler = by_id.get(handler_id)
        if endpoint is None or handler is None or endpoint.kind != "Endpoint":
            continue
        method, path = _endpoint_parts(endpoint)
        if path:
            code_endpoint_handlers.setdefault((method, path), []).append(handler)

    for endpoint in endpoint_targets:
        if endpoint.artifact_kind != "openapi":
            continue
        method, path = _endpoint_parts(endpoint)
        if not path:
            continue
        for handler in code_endpoint_handlers.get((method, path), ()):
            _add_link(links, Link(endpoint, handler, "DOC_MENTIONS", DOC_EXACT, 0.95, "openapi_endpoint_to_handler", True))


def _link_semantic_candidates(doc_nodes: list[GraphNode], code_targets: list[GraphNode], links: dict[str, Link]) -> None:
    semantic_docs = [node for node in doc_nodes if node.kind in {"Requirement", "Policy", "BusinessPolicy", "PermissionRule", "UserAction", "Screen"}]
    semantic_code = [node for node in code_targets if node.kind in {"Class", "Function", "Method", "Endpoint"}]
    for doc in semantic_docs:
        doc_terms = _meaningful_terms(f"{doc.label} {doc.chunk_text}")
        if len(doc_terms) < 2:
            continue
        for target in semantic_code:
            target_terms = _meaningful_terms(f"{target.label} {target.canonical_name}")
            shared = doc_terms & target_terms
            if len(shared) < 2:
                continue
            _add_link(links, Link(doc, target, "SEMANTIC_CANDIDATE", AMBIGUOUS, 0.2, "weak_lexical_overlap", False))


def _upsert_link(root: Path, link: Link) -> None:
    storage.upsert_edge(
        root,
        id=_link_id(link),
        src=link.source.id,
        dst=link.target.id,
        relation=link.relation,
        confidence=link.confidence,
        score=link.score,
        extractor=EXTRACTOR,
        evidence_id=storage.evidence_id_for_chunk(link.source.id),
        metadata={
            "reason": link.reason,
            "default_context_pack": link.default_context_pack,
            "source_evidence": _evidence_metadata(link.source),
            "target_evidence": _evidence_metadata(link.target),
        },
    )


def _add_link(links: dict[str, Link], link: Link) -> None:
    if link.source.id == link.target.id:
        return
    key = _link_id(link)
    current = links.get(key)
    if current is None or link.score > current.score:
        links[key] = link


def _link_id(link: Link) -> str:
    value = "\n".join([link.source.id, link.target.id, link.relation, link.reason])
    return f"link:{sha256_text(value)[:24]}"


def _is_doc_node(node: GraphNode) -> bool:
    if node.artifact_kind not in DOC_KINDS:
        return False
    return node.kind in {
        "APIReference",
        "BusinessPolicy",
        "CodeReference",
        "DatabaseEntity",
        "FormField",
        "PermissionRule",
        "Policy",
        "Requirement",
        "Screen",
        "StatusValue",
        "UserAction",
        "ValidationRule",
    }


def _is_sb_doc(node: GraphNode) -> bool:
    path = (node.artifact_path or node.path or "").lower()
    return node.artifact_kind == "pdf" or path.endswith((".sb", ".sourcebook"))


def _endpoint_parts(node: GraphNode) -> tuple[str | None, str | None]:
    method = node.metadata.get("method")
    path = node.metadata.get("path") or node.metadata.get("endpoint")
    if isinstance(method, str):
        method = method.upper()
    else:
        method = None
    if isinstance(path, str):
        return method if method in ENDPOINT_METHODS else None, _clean_endpoint_path(path)

    match = re.search(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[A-Za-z0-9_./:{}?=&%+-]+)", node.label)
    if match:
        return match.group(1).upper(), _clean_endpoint_path(match.group(2))
    bare = re.search(r"(?<![\w/])(/api/[A-Za-z0-9_./:{}?=&%+-]+)", node.label)
    if bare:
        return None, _clean_endpoint_path(bare.group(1))
    return None, None


def _clean_endpoint_path(value: str) -> str:
    return value.strip().rstrip(".,;:)]}").lower()


def _index_by_identifier(nodes: list[GraphNode], include_canonical: bool = False) -> dict[str, list[GraphNode]]:
    index: dict[str, list[GraphNode]] = {}
    for node in nodes:
        keys = {_identifier_key(node.label)}
        if include_canonical:
            keys.add(_identifier_key(node.canonical_name))
        for key in keys:
            if key:
                index.setdefault(key, []).append(node)
    return index


def _index_by_path(nodes: list[GraphNode]) -> dict[str, list[GraphNode]]:
    index: dict[str, list[GraphNode]] = {}
    for node in nodes:
        for value in (node.path, node.artifact_path, node.label, node.canonical_name):
            key = _normalize_path(value or "")
            if key:
                index.setdefault(key, []).append(node)
    return index


def _identifier_key(value: str | None) -> str:
    if not value:
        return ""
    cleaned = _symbol_reference(value)
    return cleaned.strip("`'\"").lower()


def _normalize_path(value: str) -> str:
    return value.strip().strip("`'\"").removeprefix("./").replace("\\", "/").lower()


def _doc_reference_type(node: GraphNode) -> str:
    value = node.metadata.get("reference_type") or node.metadata.get("fact_type") or ""
    return str(value)


def _symbol_reference(value: str) -> str:
    cleaned = value.strip().strip("`'\"")
    if cleaned.endswith("()"):
        cleaned = cleaned[:-2]
    if "." in cleaned and "/" not in cleaned:
        cleaned = cleaned.rsplit(".", 1)[-1]
    return cleaned


def _validation_field(node: GraphNode) -> str:
    for key in ("subject", "field", "value"):
        value = node.metadata.get(key)
        if isinstance(value, str) and _looks_like_identifier(value):
            return value
    terms = node.metadata.get("normalized_terms")
    if isinstance(terms, list):
        for term in terms:
            if isinstance(term, str) and _looks_like_identifier(term) and term not in {"validation", "input"}:
                return term
    for match in re.findall(r"`([^`\n]+)`", node.chunk_text):
        if _looks_like_identifier(match):
            return match
    return ""


def _looks_like_validation_code(node: GraphNode) -> bool:
    text = f"{node.label} {node.canonical_name} {node.chunk_text}".lower()
    return any(token in text for token in VALIDATION_TOKENS)


def _contains_identifier(text: str, key: str) -> bool:
    if not key:
        return False
    return any(_identifier_key(token) == key for token in re.findall(r"[A-Za-z_$][\w$.-]*", text))


def _looks_like_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_$][\w$.-]*", value)) and not _is_generic_single_term(value)


def _is_generic_single_term(value: str) -> bool:
    terms = _split_terms(value)
    return len(terms) == 1 and next(iter(terms), "") in GENERIC_TERMS


def _meaningful_terms(value: str) -> set[str]:
    return {term for term in _split_terms(value) if len(term) >= 3 and term not in GENERIC_TERMS}


def _split_terms(value: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", spaced)}


def _evidence_metadata(node: GraphNode) -> dict[str, Any]:
    return {
        "node_id": node.id,
        "path": node.path or node.artifact_path,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "page": node.page,
        "quote": _short_quote(node.chunk_text or node.label),
    }


def _short_quote(value: str, limit: int = 400) -> str:
    compact = " ".join(value.strip().split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def _json_load(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}
