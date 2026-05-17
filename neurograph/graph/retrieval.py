"""Evidence-first local retrieval."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Iterable

from neurograph import storage
from neurograph.graph.schema import Chunk, SearchHit
from neurograph.utils.paths import db_path


TOKEN_RE = re.compile(r"[A-Za-z0-9_.$/-]+")
ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[A-Za-z0-9_./:{}?=&%+-]+)", re.IGNORECASE)
BARE_API_RE = re.compile(r"(?<![\w/])(/api/[A-Za-z0-9_./:{}?=&%+-]+)")
FILE_PATH_RE = re.compile(
    r"(?<!://)(?:\.{0,2}/)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]+"
    r"|(?<!://)\b[A-Za-z0-9_.-]+\.(?:py|ts|tsx|js|jsx|md|markdown|yaml|yml|json|toml|sql|graphql|go|rs|java|swift|rb|php)\b"
)
STATUS_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
IDENTIFIER_RE = re.compile(r"[A-Za-z_$][\w$]*(?:[._-][A-Za-z_$][\w$]*)*")

DOC_KINDS = {"markdown", "pdf", "sb"}
CODE_KINDS = {"code", "openapi", "sql", "config"}
SEMANTIC_RELATION = "SEMANTIC_CANDIDATE"
DEFAULT_EDGE_TYPES = {
    "CONTAINS",
    "DEFINES",
    "DOC_EXPLAINS",
    "DOC_MENTIONS",
    "IMPLEMENTS",
    "IMPORTS",
    "QUERY_TOUCHES_TABLE",
    "REFERENCES",
    "ROUTE_TO_HANDLER",
    "SB_DEFINES_REQUIREMENT",
    "SB_DEFINES_VALIDATION",
    "SB_REFERENCES_API",
    "SB_REFERENCES_STATUS",
}
GENERIC_TERMS = {
    "api",
    "app",
    "data",
    "delete",
    "flow",
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
CONFIDENCE_WEIGHT = {
    "EXACT_COMPILER": 30.0,
    "EXACT_STATIC": 22.0,
    "DOC_EXACT": 16.0,
    "DOC_INFERRED": 8.0,
    "AMBIGUOUS": -18.0,
}
EXTRACTOR_WEIGHT = {
    "scip": 24.0,
    "code_fast_graph": 14.0,
    "code_document_linker": 12.0,
    "markdown": 5.0,
    "pdf_fast_text": 5.0,
}


@dataclass(frozen=True)
class RetrievalNode:
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
    indexed_at: datetime | None


@dataclass(frozen=True)
class SeedNode:
    node: RetrievalNode
    reasons: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class RetrievalEdge:
    id: str
    src: str
    dst: str
    relation: str
    confidence: str | None
    score: float | None
    extractor: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TraversalHit:
    node_id: str
    distance: int
    path: tuple[RetrievalEdge, ...] = ()


@dataclass(frozen=True)
class RetrievalCandidate:
    node: RetrievalNode
    score: float
    reasons: tuple[str, ...]
    distance: int | None = None
    path: tuple[RetrievalEdge, ...] = ()


@dataclass(frozen=True)
class EvidenceItem:
    node_id: str
    node_kind: str
    label: str
    path: str
    quote: str
    location: str
    confidence: str | None
    score: float
    reasons: tuple[str, ...]
    graph_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalResult:
    task: str
    intent: str
    seeds: tuple[SeedNode, ...]
    candidates: tuple[RetrievalCandidate, ...]
    evidence: tuple[EvidenceItem, ...]
    unknowns: tuple[str, ...] = ()


@dataclass
class _CandidateState:
    node: RetrievalNode
    base_score: float = 0.0
    reasons: set[str] = field(default_factory=set)
    distance: int | None = None
    path: tuple[RetrievalEdge, ...] = ()


def tokenize(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value) if len(token) > 1]


def infer_intent(task: str) -> str:
    """Infer a broad retrieval intent from a user task."""

    text = task.lower()
    if re.search(r"\b(why|reason|rationale|explain|what caused|because)\b", text):
        return "why"
    if re.search(r"\b(conflict|inconsistent|mismatch|contradict|disagree|drift)\b", text):
        return "conflict"
    if re.search(r"\b(impact|affected|blast radius|change|modify|rename|remove|break|migration)\b", text):
        return "impact"
    if re.search(r"\b(implement|add|build|create|fix|wire|support|handle)\b", text):
        return "implementation"
    return "auto"


def find_seed_nodes(root: Path, task: str) -> list[SeedNode]:
    """Find high-precision seed nodes by exact project term matching."""

    nodes = _load_nodes(root)
    query = _query_terms(task)
    seeds: dict[str, SeedNode] = {}

    for node in nodes:
        reasons: list[str] = []
        method, endpoint_path = _endpoint_parts(node)
        if endpoint_path and node.kind in {"Endpoint", "APIReference"}:
            endpoint_matched = False
            method_matched = False
            for expected_method, expected_path in sorted(query["endpoints"], key=lambda item: (item[0] is None, item[0] or "", item[1])):
                if endpoint_path == expected_path and (expected_method is None or method == expected_method):
                    endpoint_matched = True
                    if expected_method is not None and method == expected_method:
                        method_matched = True
            if endpoint_matched:
                reasons.append("exact_endpoint")
            if method_matched:
                reasons.append("exact_endpoint_method")

        node_path = _normalize_path(node.path or node.artifact_path or node.label)
        if node_path and node_path in query["paths"]:
            reasons.append("exact_path")

        label_key = _identifier_key(node.label)
        canonical_key = _identifier_key(node.canonical_name)
        keys = {key for key in (label_key, canonical_key) if key}
        if node.kind in {"Class", "Function", "Method", "Symbol"} and keys & query["symbols"]:
            reasons.append("exact_symbol")
        if node.kind in {"Column", "FormField"} and keys & query["fields"]:
            reasons.append("exact_field")
        if node.kind == "Table" and keys & query["tables"]:
            reasons.append("exact_table")
        if node.kind == "ConfigKey" and keys & query["config_keys"]:
            reasons.append("exact_config_key")
        if node.kind in {"StatusValue", "ConfigKey", "Column"} and keys & query["statuses"]:
            reasons.append("exact_status_value")

        if reasons:
            score = 100.0 + max(_node_confidence_weight(node), 0)
            if "exact_endpoint_method" in reasons:
                score += 8.0
            seeds[node.id] = SeedNode(node=node, reasons=tuple(sorted(set(reasons))), score=score)

    return sorted(seeds.values(), key=lambda seed: (-seed.score, seed.node.path or "", seed.node.start_line or 0, seed.node.label))


def traverse_from_seeds(
    root: Path,
    seeds: Iterable[SeedNode | RetrievalNode | str],
    edge_types: set[str] | None = None,
    max_depth: int = 3,
    include_semantic: bool = False,
) -> list[TraversalHit]:
    """Traverse typed graph edges from seed nodes with deterministic BFS."""

    allowed = set(edge_types or DEFAULT_EDGE_TYPES)
    if include_semantic:
        allowed.add(SEMANTIC_RELATION)
    nodes_by_id = {node.id: node for node in _load_nodes(root)}
    edges_by_node = _edges_by_node(root, allowed)
    seed_ids = [_seed_id(seed) for seed in seeds]
    seed_ids = [node_id for node_id in seed_ids if node_id in nodes_by_id]

    hits: dict[str, TraversalHit] = {}
    queue: deque[tuple[str, int, tuple[RetrievalEdge, ...]]] = deque()
    for node_id in sorted(set(seed_ids)):
        hits[node_id] = TraversalHit(node_id=node_id, distance=0, path=())
        queue.append((node_id, 0, ()))

    while queue:
        node_id, distance, path = queue.popleft()
        if distance >= max_depth:
            continue
        for edge in sorted(edges_by_node.get(node_id, ()), key=_edge_sort_key):
            if edge.relation == SEMANTIC_RELATION and not include_semantic:
                continue
            next_id = edge.dst if edge.src == node_id else edge.src
            if next_id not in nodes_by_id:
                continue
            next_path = (*path, edge)
            next_distance = distance + 1
            current = hits.get(next_id)
            if current is not None and _traversal_sort_key(current) <= _traversal_sort_key(TraversalHit(next_id, next_distance, next_path)):
                continue
            hit = TraversalHit(node_id=next_id, distance=next_distance, path=next_path)
            hits[next_id] = hit
            queue.append((next_id, next_distance, next_path))

    return sorted(hits.values(), key=_traversal_sort_key)


def rank_candidates(root: Path, candidates: Iterable[RetrievalCandidate]) -> list[RetrievalCandidate]:
    """Rank candidates deterministically using exactness, graph quality, and freshness."""

    candidate_list = list(candidates)
    freshness = _freshness_rank(candidate_list)
    ranked: list[RetrievalCandidate] = []
    for candidate in candidate_list:
        score = candidate.score + _node_confidence_weight(candidate.node) + freshness.get(candidate.node.id, 0.0)
        if candidate.distance is not None:
            score += max(0.0, 24.0 - (candidate.distance * 5.0))
            score += _path_weight(candidate.path)
        if any(edge.relation == SEMANTIC_RELATION for edge in candidate.path):
            score -= 30.0
        ranked.append(
            RetrievalCandidate(
                node=candidate.node,
                score=round(score, 4),
                reasons=tuple(sorted(set(candidate.reasons))),
                distance=candidate.distance,
                path=candidate.path,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.score,
            item.distance if item.distance is not None else 99,
            item.node.artifact_path or item.node.path or "",
            item.node.start_line or 0,
            item.node.page or 0,
            item.node.kind,
            item.node.label,
        )
    )
    return ranked


def collect_evidence(root: Path, candidates: Iterable[RetrievalCandidate], limit: int = 8) -> list[EvidenceItem]:
    """Collect compact evidence snippets for ranked candidates."""

    evidence: list[EvidenceItem] = []
    seen: set[tuple[str, int | None, int | None, int | None, str]] = set()

    for candidate in candidates:
        node = candidate.node
        quote = _short_quote(node.chunk_text or node.metadata.get("quote") or node.label)
        if not quote:
            continue
        path = node.path or node.artifact_path or ""
        key = (path, node.start_line, node.end_line, node.page, quote)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            EvidenceItem(
                node_id=node.id,
                node_kind=node.kind,
                label=node.label,
                path=path,
                quote=quote,
                location=_location(path, node.start_line, node.end_line, node.page),
                confidence=node.confidence,
                score=candidate.score,
                reasons=candidate.reasons,
                graph_path=tuple(edge.relation for edge in candidate.path),
            )
        )
        if len(evidence) >= limit:
            break

    return evidence


def produce_retrieval_result(
    root: Path,
    task: str,
    *,
    limit: int = 8,
    max_depth: int = 3,
    include_semantic: bool = False,
    allow_fts: bool = True,
) -> RetrievalResult:
    """Run exact, text, graph, and ranking retrieval for one task."""

    intent = infer_intent(task)
    nodes_by_id = {node.id: node for node in _load_nodes(root)}
    seeds = find_seed_nodes(root, task)
    text_candidates = _text_seed_candidates(root, task, allow_fts=allow_fts)

    states: dict[str, _CandidateState] = {}
    for seed in seeds:
        _merge_state(states, seed.node, seed.score, seed.reasons, None, ())
    for candidate in text_candidates:
        _merge_state(states, candidate.node, candidate.score, candidate.reasons, candidate.distance, candidate.path)

    traversal_seeds: list[SeedNode | RetrievalNode] = [*seeds, *(candidate.node for candidate in text_candidates[:10])]
    if traversal_seeds:
        for hit in traverse_from_seeds(root, traversal_seeds, _edge_types_for_intent(intent), max_depth=max_depth, include_semantic=include_semantic):
            node = nodes_by_id.get(hit.node_id)
            if node is None:
                continue
            base = max(0.0, 62.0 - (hit.distance * 10.0))
            if hit.path and hit.path[-1].relation == SEMANTIC_RELATION:
                base = min(base, 8.0)
            _merge_state(states, node, base, ("graph_traversal",), hit.distance, hit.path)

    ranked = rank_candidates(
        root,
        (
            RetrievalCandidate(
                node=state.node,
                score=state.base_score,
                reasons=tuple(state.reasons),
                distance=state.distance,
                path=state.path,
            )
            for state in states.values()
        ),
    )
    ranked = ranked[: max(limit * 3, limit)]
    evidence = collect_evidence(root, ranked, limit=limit)
    unknowns = []
    if not seeds:
        unknowns.append("No exact path, symbol, endpoint, table, config key, or status seed matched the task.")
    if not include_semantic:
        unknowns.append("Semantic candidates are an extension point and are disabled by default in v0.1 retrieval.")

    return RetrievalResult(
        task=task,
        intent=intent,
        seeds=tuple(seeds),
        candidates=tuple(ranked),
        evidence=tuple(evidence),
        unknowns=tuple(unknowns),
    )


def search(root: Path, query: str, limit: int = 8) -> list[SearchHit]:
    """Backward-compatible search API returning evidence chunks."""

    result = produce_retrieval_result(root, query, limit=limit)
    hits: list[SearchHit] = []
    for item in result.evidence:
        chunk = Chunk(
            id=item.node_id,
            file_path=item.path,
            kind=item.node_kind,
            text=item.quote,
            evidence=item.location,
            symbol=item.label if item.node_kind in {"Class", "Function", "Method", "Symbol"} else None,
            heading=item.label if item.node_kind in {"Document", "Section"} else None,
        )
        hits.append(SearchHit(chunk=chunk, score=item.score, reason="+".join(item.reasons)))
    return hits


def summarize_chunk(chunk: Chunk, max_chars: int = 900) -> str:
    text = chunk.text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _load_nodes(root: Path) -> list[RetrievalNode]:
    if not db_path(root).exists():
        return []
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
                c.text,
                a.indexed_at
            FROM nodes n
            LEFT JOIN artifacts a ON a.id = n.artifact_id
            LEFT JOIN chunks c ON c.id = n.id
            ORDER BY n.id
            """
        ).fetchall()
    return [
        RetrievalNode(
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
            indexed_at=row[13] if isinstance(row[13], datetime) else None,
        )
        for row in rows
    ]


def _edges_by_node(root: Path, allowed: set[str]) -> dict[str, list[RetrievalEdge]]:
    if not db_path(root).exists() or not allowed:
        return {}
    with storage.connect(root) as con:
        rows = con.execute(
            """
            SELECT id, src, dst, relation, confidence, score, extractor, metadata
            FROM edges
            WHERE relation IN ({})
            """.format(", ".join("?" for _ in allowed)),
            sorted(allowed),
        ).fetchall()
    result: dict[str, list[RetrievalEdge]] = {}
    for row in rows:
        edge = RetrievalEdge(
            id=str(row[0]),
            src=str(row[1]),
            dst=str(row[2]),
            relation=str(row[3]),
            confidence=row[4],
            score=float(row[5]) if row[5] is not None else None,
            extractor=row[6],
            metadata=_json_load(row[7]),
        )
        result.setdefault(edge.src, []).append(edge)
        result.setdefault(edge.dst, []).append(edge)
    return result


def _text_seed_candidates(root: Path, task: str, limit: int = 20, *, allow_fts: bool = True) -> list[RetrievalCandidate]:
    nodes_by_id = {node.id: node for node in _load_nodes(root)}
    candidates: dict[str, RetrievalCandidate] = {}

    for hit in storage.search_text(root, task, limit=limit, allow_fts=allow_fts):
        node = nodes_by_id.get(str(hit["id"]))
        if node is None or node.artifact_kind not in DOC_KINDS:
            continue
        score = 38.0 + min(float(hit.get("score", 0.0)), 10.0)
        candidates[node.id] = RetrievalCandidate(
            node=node,
            score=score,
            reasons=(f"text_search:{hit.get('search_method', 'unknown')}",),
        )

    for node in _evidence_quote_nodes(root, task, nodes_by_id, limit=limit):
        candidates.setdefault(
            node.id,
            RetrievalCandidate(node=node, score=34.0, reasons=("evidence_quote",)),
        )

    return sorted(candidates.values(), key=lambda item: (-item.score, item.node.path or "", item.node.start_line or 0, item.node.label))


def _evidence_quote_nodes(root: Path, task: str, nodes_by_id: dict[str, RetrievalNode], limit: int = 20) -> list[RetrievalNode]:
    tokens = _significant_query_tokens(task)
    if not tokens:
        return []
    patterns = [f"%{token}%" for token in tokens[:6]]
    clauses = " OR ".join("lower(e.quote) LIKE ?" for _ in patterns)
    if not db_path(root).exists():
        return []
    with storage.connect(root) as con:
        rows = con.execute(
            f"""
            SELECT n.id
            FROM evidence e
            JOIN artifacts a ON a.id = e.artifact_id
            JOIN nodes n ON n.artifact_id = e.artifact_id
                AND ((n.start_line = e.start_line AND n.end_line = e.end_line) OR n.page = e.page)
            WHERE a.kind IN ('markdown', 'pdf', 'sb')
              AND ({clauses})
            ORDER BY a.path, n.start_line, n.page, n.id
            LIMIT ?
            """,
            [*patterns, limit],
        ).fetchall()
    return [nodes_by_id[row[0]] for row in rows if row[0] in nodes_by_id]


def _query_terms(task: str) -> dict[str, set[Any]]:
    endpoints: set[tuple[str | None, str]] = set()
    for method, path in ENDPOINT_RE.findall(task):
        endpoints.add((method.upper(), _clean_endpoint_path(path)))
    for path in BARE_API_RE.findall(task):
        endpoints.add((None, _clean_endpoint_path(path)))

    paths = {_normalize_path(match) for match in FILE_PATH_RE.findall(task)}
    statuses = {_identifier_key(match) for match in STATUS_RE.findall(task) if match not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "API", "HTTP", "PDF"}}
    table_terms = {
        _identifier_key(value)
        for groups in re.findall(r"\b([A-Za-z_$][\w$]*)\s+table\b|\btable\s+([A-Za-z_$][\w$]*)", task, re.IGNORECASE)
        for value in groups
        if value
    }
    fields = {
        _identifier_key(value)
        for groups in re.findall(r"`([^`\n]+)`|(?:field|column)\s+([A-Za-z_$][\w$.-]*)|([A-Za-z_$][\w$.-]*)\s+(?:field|column)", task, re.IGNORECASE)
        for value in groups
        if value
    }
    config_keys = {
        _identifier_key(value)
        for groups in re.findall(r"`([^`\n]+)`|([A-Za-z_$][\w$]*(?:[._-][A-Za-z_$][\w$]*)+)", task)
        for value in groups
        if value
    }

    symbols: set[str] = set()
    for token in IDENTIFIER_RE.findall(task):
        key = _identifier_key(token)
        if key and not _is_generic_single_term(token):
            symbols.add(key)

    fields |= {key for key in config_keys if key}
    return {
        "endpoints": endpoints,
        "paths": {path for path in paths if path},
        "statuses": {status for status in statuses if status},
        "tables": {term for term in table_terms if term},
        "fields": {field for field in fields if field},
        "config_keys": {key for key in config_keys if key},
        "symbols": symbols,
    }


def _edge_types_for_intent(intent: str) -> set[str]:
    if intent == "why":
        return DEFAULT_EDGE_TYPES | {"DOC_EXPLAINS"}
    if intent == "conflict":
        return DEFAULT_EDGE_TYPES | {"DOC_MENTIONS", "DOC_EXPLAINS"}
    return set(DEFAULT_EDGE_TYPES)


def _merge_state(
    states: dict[str, _CandidateState],
    node: RetrievalNode,
    score: float,
    reasons: Iterable[str],
    distance: int | None,
    path: tuple[RetrievalEdge, ...],
) -> None:
    state = states.get(node.id)
    if state is None:
        state = _CandidateState(node=node)
        states[node.id] = state
    state.base_score = max(state.base_score, score)
    state.reasons.update(reasons)
    if distance is not None and (state.distance is None or distance < state.distance or (distance == state.distance and _path_weight(path) > _path_weight(state.path))):
        state.distance = distance
        state.path = path


def _seed_id(seed: SeedNode | RetrievalNode | str) -> str:
    if isinstance(seed, SeedNode):
        return seed.node.id
    if isinstance(seed, RetrievalNode):
        return seed.id
    return seed


def _endpoint_parts(node: RetrievalNode) -> tuple[str | None, str | None]:
    method = node.metadata.get("method")
    path = node.metadata.get("path") or node.metadata.get("endpoint")
    method_text = method.upper() if isinstance(method, str) else None
    if isinstance(path, str) and path:
        if method_text is None:
            match = ENDPOINT_RE.search(node.label)
            if match and _clean_endpoint_path(match.group(2)) == _clean_endpoint_path(path):
                method_text = match.group(1).upper()
        return method_text, _clean_endpoint_path(path)
    match = ENDPOINT_RE.search(node.label)
    if match:
        return match.group(1).upper(), _clean_endpoint_path(match.group(2))
    bare = BARE_API_RE.search(node.label)
    if bare:
        return None, _clean_endpoint_path(bare.group(1))
    return None, None


def _clean_endpoint_path(value: str) -> str:
    return value.strip().rstrip(".,;:)]}").lower()


def _identifier_key(value: str | None) -> str:
    if not value:
        return ""
    cleaned = str(value).strip().strip("`'\"")
    if cleaned.endswith("()"):
        cleaned = cleaned[:-2]
    return cleaned.lower()


def _normalize_path(value: str | None) -> str:
    return (value or "").strip().strip("`'\"").removeprefix("./").replace("\\", "/").lower()


def _significant_query_tokens(task: str) -> list[str]:
    return [token for token in tokenize(task) if len(token) >= 3 and token not in GENERIC_TERMS]


def _is_generic_single_term(value: str) -> bool:
    terms = re.findall(r"[A-Za-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value))
    return len(terms) == 1 and terms[0].lower() in GENERIC_TERMS


def _node_confidence_weight(node: RetrievalNode) -> float:
    return CONFIDENCE_WEIGHT.get(str(node.confidence or ""), 0.0)


def _path_weight(path: tuple[RetrievalEdge, ...]) -> float:
    if not path:
        return 0.0
    total = 0.0
    for edge in path:
        total += CONFIDENCE_WEIGHT.get(str(edge.confidence or ""), 0.0) / 4.0
        total += EXTRACTOR_WEIGHT.get(str(edge.extractor or ""), 0.0) / 3.0
        if edge.relation == SEMANTIC_RELATION:
            total -= 20.0
    return total


def _freshness_rank(candidates: list[RetrievalCandidate]) -> dict[str, float]:
    timestamps = sorted({candidate.node.indexed_at for candidate in candidates if candidate.node.indexed_at}, reverse=True)
    if not timestamps:
        return {}
    rank = {timestamp: max(0.0, 4.0 - (index * 0.25)) for index, timestamp in enumerate(timestamps)}
    return {candidate.node.id: rank.get(candidate.node.indexed_at, 0.0) for candidate in candidates}


def _edge_sort_key(edge: RetrievalEdge) -> tuple[float, str, str, str]:
    return (-(_path_weight((edge,)) + float(edge.score or 0.0)), edge.relation, edge.src, edge.dst)


def _traversal_sort_key(hit: TraversalHit) -> tuple[int, float, str]:
    return (hit.distance, -_path_weight(hit.path), hit.node_id)


def _location(path: str, start_line: int | None, end_line: int | None, page: int | None) -> str:
    if page is not None:
        return f"{path}:page {page}"
    if start_line is not None and end_line is not None:
        if start_line == end_line:
            return f"{path}:{start_line}"
        return f"{path}:{start_line}-{end_line}"
    return path


def _short_quote(value: Any, limit: int = 900) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _json_load(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}
