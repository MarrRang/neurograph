"""Deterministic Markdown indexing for evidence-backed document nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

from neurograph.indexer.sb_facts import SBFact, extract_sb_facts, fact_id
from neurograph.utils.hashing import sha256_text


DOC_EXACT = "DOC_EXACT"
DOC_INFERRED = "DOC_INFERRED"
AMBIGUOUS = "AMBIGUOUS"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*([A-Za-z0-9_+.-]*)?.*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[A-Za-z0-9_./:{}?=&%+-]+)")
BARE_API_RE = re.compile(r"(?<![\w/])(/api/[A-Za-z0-9_./:{}?=&%+-]+)")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
FILE_PATH_RE = re.compile(
    r"(?<!://)(?:\.{0,2}/)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]+"
    r"|(?<!://)\b[A-Za-z0-9_.-]+\.(?:py|ts|tsx|js|jsx|md|markdown|yaml|yml|json|toml|sql|graphql|go|rs|java|swift|rb|php)\b"
)
FUNCTION_RE = re.compile(r"\b([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s*\(")
CONFIG_KEY_RE = re.compile(r"\b([a-z][A-Za-z0-9_-]*(?:[._-][A-Za-z0-9_-]+)+)\b")
STATUS_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,}|[a-z]+(?:-[a-z0-9]+)+)\b")
PASCAL_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)+)\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
EXPLAINS_RE = re.compile(r"\b(is|are|means|refers to|defines|describes|explains|represents)\b", re.IGNORECASE)
VALIDATION_RE = re.compile(
    r"\b("
    r"must be valid|must match|must include|must contain|required field|required when|"
    r"cannot be empty|must not exceed|at least|at most|minimum|max(?:imum)?|"
    r"invalid if|reject(?:s|ed)? when|validate|validation"
    r")\b",
    re.IGNORECASE,
)
ERROR_RE = re.compile(
    r"\b(error(?: message)?|throws?|raises?|returns?)\b\s*(?:[:\-]\s*)?(?:`([^`]+)`|\"([^\"]+)\"|'([^']+)'|([A-Z][^.!?]{3,}))?",
    re.IGNORECASE,
)
POLICY_RE = re.compile(r"\b(do not|never|only|no external|must not|read-only|local-first)\b", re.IGNORECASE)
REQUIREMENT_RE = re.compile(r"\b(must|shall|should|required|needs? to|has to|have to)\b", re.IGNORECASE)


@dataclass(frozen=True)
class MarkdownEvidence:
    source_path: str
    start_line: int
    end_line: int
    quote: str
    confidence: str


@dataclass(frozen=True)
class MarkdownNode:
    id: str
    kind: str
    label: str
    canonical_name: str
    evidence: MarkdownEvidence
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MarkdownRelation:
    source: str
    target: str
    relation: str
    evidence: MarkdownEvidence
    confidence: str


@dataclass(frozen=True)
class MarkdownDocument:
    source_path: str
    title: str
    nodes: tuple[MarkdownNode, ...]
    relations: tuple[MarkdownRelation, ...]


@dataclass(frozen=True)
class MarkdownSection:
    heading: str | None
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class _Block:
    kind: str
    start_line: int
    end_line: int
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


def index_markdown(path: Path, source_path: str | None = None) -> MarkdownDocument:
    """Parse Markdown into deterministic evidence-backed document nodes."""

    source = source_path or path.as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    blocks = _parse_blocks(lines)
    heading_blocks = [block for block in blocks if block.kind == "heading"]
    title_block = next((block for block in heading_blocks if block.metadata.get("level") == 1), None)
    title = _clean_heading(title_block.text) if title_block else _fallback_title(path, lines)
    document_line = title_block.start_line if title_block else _first_nonblank_line(lines)
    document_quote = lines[document_line - 1].strip() if lines and 0 < document_line <= len(lines) else title

    nodes: list[MarkdownNode] = []
    relations: list[MarkdownRelation] = []
    seen: set[tuple[str, str, int, int, str]] = set()

    document_node = _make_node(
        source,
        "Document",
        title,
        document_line,
        document_line,
        document_quote,
        DOC_EXACT,
        {"title": title},
    )
    nodes.append(document_node)

    section_nodes: list[tuple[_Block, MarkdownNode]] = []
    for index, block in enumerate(heading_blocks):
        next_heading = heading_blocks[index + 1] if index + 1 < len(heading_blocks) else None
        section_end = (next_heading.start_line - 1) if next_heading else max(block.end_line, len(lines))
        quote = "\n".join(lines[block.start_line - 1 : section_end]).strip() or block.text.strip()
        node = _make_node(
            source,
            "Section",
            _clean_heading(block.text),
            block.start_line,
            section_end,
            quote,
            DOC_EXACT,
            {"level": block.metadata["level"]},
        )
        nodes.append(node)
        section_nodes.append((block, node))
        relations.append(_relation(document_node.id, node.id, "CONTAINS", node.evidence, DOC_EXACT))

    parent_blocks = [block for block in blocks if block.kind in {"paragraph", "table", "code_block"}]
    for block in parent_blocks:
        node_kind = {
            "paragraph": "Paragraph",
            "table": "Table",
            "code_block": "CodeBlock",
        }[block.kind]
        label = _block_label(block)
        confidence = DOC_EXACT if node_kind in {"Table", "CodeBlock"} else DOC_INFERRED
        node = _make_node(
            source,
            node_kind,
            label,
            block.start_line,
            block.end_line,
            block.text,
            confidence,
            block.metadata,
        )
        nodes.append(node)
        relations.append(_relation(_section_parent(section_nodes, block) or document_node.id, node.id, "CONTAINS", node.evidence, confidence))

        for extracted in _extract_semantic_nodes(source, block):
            key = (
                extracted.kind,
                extracted.canonical_name,
                extracted.evidence.start_line,
                extracted.evidence.end_line,
                str(extracted.metadata.get("reference_type", "")),
            )
            if key not in seen:
                nodes.append(extracted)
                seen.add(key)
            relation = _mention_relation(block.text, extracted)
            relations.append(_relation(node.id, extracted.id, relation, extracted.evidence, extracted.evidence.confidence))

    return MarkdownDocument(source_path=source, title=title, nodes=tuple(nodes), relations=tuple(relations))


def parse_markdown(path: Path) -> tuple[str | None, list[MarkdownSection]]:
    """Compatibility wrapper returning section chunks for Context Packs."""

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    document = index_markdown(path, path.as_posix())
    sections: list[MarkdownSection] = []
    section_nodes = [node for node in document.nodes if node.kind == "Section"]
    for node in section_nodes:
        start = node.evidence.start_line
        end = node.evidence.end_line
        sections.append(
            MarkdownSection(
                heading=node.label,
                start_line=start,
                end_line=end,
                text="\n".join(lines[start - 1 : end]).strip(),
            )
        )
    if not sections and text.strip():
        sections.append(MarkdownSection(heading=None, start_line=1, end_line=len(lines), text=text.strip()))
    return document.title, sections


def _parse_blocks(lines: list[str]) -> list[_Block]:
    blocks: list[_Block] = []
    paragraph_start: int | None = None
    paragraph_lines: list[str] = []
    index = 0

    def flush_paragraph(end_line: int) -> None:
        nonlocal paragraph_start, paragraph_lines
        body = "\n".join(paragraph_lines).strip()
        if body and paragraph_start is not None:
            blocks.append(_Block("paragraph", paragraph_start, end_line, body, {"links": _extract_links(body)}))
        paragraph_start = None
        paragraph_lines = []

    while index < len(lines):
        line_number = index + 1
        line = lines[index]
        stripped = line.strip()

        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_paragraph(line_number - 1)
            blocks.append(
                _Block("heading", line_number, line_number, line.strip(), {"level": len(heading_match.group(1))})
            )
            index += 1
            continue

        fence_match = FENCE_RE.match(line)
        if fence_match:
            flush_paragraph(line_number - 1)
            fence = fence_match.group(1)
            language = fence_match.group(2) or ""
            start = line_number
            collected = [line]
            index += 1
            while index < len(lines):
                collected.append(lines[index])
                if lines[index].strip().startswith(fence):
                    break
                index += 1
            end = min(index + 1, len(lines))
            blocks.append(_Block("code_block", start, end, "\n".join(collected).strip(), {"language": language}))
            index += 1
            continue

        if _is_table_start(lines, index):
            flush_paragraph(line_number - 1)
            start = line_number
            collected = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                collected.append(lines[index])
                index += 1
            blocks.append(_Block("table", start, start + len(collected) - 1, "\n".join(collected).strip(), {}))
            continue

        if not stripped:
            flush_paragraph(line_number - 1)
            index += 1
            continue

        if paragraph_start is None:
            paragraph_start = line_number
        paragraph_lines.append(line)
        index += 1

    flush_paragraph(len(lines))
    return blocks


def _extract_semantic_nodes(source: str, block: _Block) -> list[MarkdownNode]:
    nodes: list[MarkdownNode] = []
    text = block.text
    for fact in extract_sb_facts(text, start_line=block.start_line, end_line=block.end_line):
        nodes.append(_node_from_fact(source, fact, block))

    for method, endpoint in ENDPOINT_RE.findall(text):
        label = f"{method} {endpoint}"
        nodes.append(_mention_node(source, "APIReference", label, block, DOC_EXACT, {"method": method, "endpoint": endpoint}))
    for endpoint in BARE_API_RE.findall(text):
        nodes.append(_mention_node(source, "APIReference", endpoint, block, DOC_EXACT, {"endpoint": endpoint}))

    for link_text, href in _extract_links(text):
        metadata = {"link_text": link_text, "href": href}
        if _looks_like_api(href):
            nodes.append(_mention_node(source, "APIReference", href, block, DOC_EXACT, metadata))
        elif _looks_like_file_path(href):
            nodes.append(_mention_node(source, "CodeReference", href, block, DOC_EXACT, {**metadata, "reference_type": "file_path"}))
        else:
            nodes.append(_mention_node(source, "CodeReference", href, block, DOC_EXACT, {**metadata, "reference_type": "link"}))

    for value in _backticked_values(text):
        if _looks_like_file_path(value):
            nodes.append(_mention_node(source, "CodeReference", value, block, DOC_EXACT, {"reference_type": "file_path"}))
        elif _looks_like_function(value):
            nodes.append(_mention_node(source, "CodeReference", value, block, DOC_EXACT, {"reference_type": "function"}))
        elif _looks_like_config_key(value):
            nodes.append(_mention_node(source, "CodeReference", value, block, DOC_EXACT, {"reference_type": "config_key"}))
        elif _looks_like_status(value):
            nodes.append(_mention_node(source, "StatusValue", value, block, DOC_EXACT, {"reference_type": "status_value"}))

    for file_path in FILE_PATH_RE.findall(text):
        if "://" not in file_path:
            nodes.append(_mention_node(source, "CodeReference", file_path, block, DOC_EXACT, {"reference_type": "file_path"}))

    for function in FUNCTION_RE.findall(text):
        nodes.append(_mention_node(source, "CodeReference", f"{function}()", block, DOC_EXACT, {"reference_type": "function"}))

    for key in CONFIG_KEY_RE.findall(text):
        if "." in key or "_" in key or "-" in key:
            nodes.append(_mention_node(source, "CodeReference", key, block, DOC_EXACT, {"reference_type": "config_key"}))

    for status in STATUS_RE.findall(text):
        if _looks_like_status(status):
            nodes.append(_mention_node(source, "StatusValue", status, block, DOC_EXACT, {"reference_type": "status_value"}))

    for sentence in _sentences(text):
        sentence_block = _sentence_block(block, sentence)
        if ERROR_RE.search(sentence):
            nodes.append(_mention_node(source, "ErrorMessage", _error_label(sentence), sentence_block, DOC_INFERRED, {}))
        elif VALIDATION_RE.search(sentence):
            nodes.append(_mention_node(source, "ValidationRule", _short_label(sentence), sentence_block, DOC_INFERRED, {}))
        elif POLICY_RE.search(sentence):
            nodes.append(_mention_node(source, "Policy", _short_label(sentence), sentence_block, DOC_INFERRED, {}))
        elif REQUIREMENT_RE.search(sentence):
            nodes.append(_mention_node(source, "Requirement", _short_label(sentence), sentence_block, DOC_INFERRED, {}))

    for candidate in PASCAL_RE.findall(text):
        if candidate not in {"HTTP", "JSON", "YAML", "SQL"}:
            nodes.append(_mention_node(source, "CodeReference", candidate, block, AMBIGUOUS, {"reference_type": "class_candidate"}))

    return _dedupe_nodes(nodes)


def _node_from_fact(source: str, fact: SBFact, fallback_block: _Block) -> MarkdownNode:
    kind = _fact_kind(fact.type)
    start_line = fact.start_line or fallback_block.start_line
    end_line = fact.end_line or fallback_block.end_line
    label = _fact_label(fact)
    evidence = MarkdownEvidence(
        source_path=source,
        start_line=start_line,
        end_line=end_line,
        quote=fact.evidence_quote,
        confidence=fact.confidence,
    )
    return MarkdownNode(
        id=fact_id(source, fact),
        kind=kind,
        label=label,
        canonical_name=label,
        evidence=evidence,
        metadata={
            "fact_type": fact.type,
            "subject": fact.subject,
            "property": fact.property,
            "value": fact.value,
            "normalized_terms": list(fact.normalized_terms),
        },
    )


def _fact_kind(fact_type: str) -> str:
    return {
        "screen": "Screen",
        "user_action": "UserAction",
        "requirement": "Requirement",
        "validation_rule": "ValidationRule",
        "business_policy": "BusinessPolicy",
        "api_reference": "APIReference",
        "error_message": "ErrorMessage",
        "status_value": "StatusValue",
        "form_field": "FormField",
        "permission_rule": "PermissionRule",
        "database_entity": "DatabaseEntity",
        "external_integration": "ExternalIntegration",
    }.get(fact_type, "Requirement")


def _fact_label(fact: SBFact) -> str:
    if fact.type == "api_reference":
        return f"{fact.value} {fact.subject}" if fact.property == "method" else fact.subject
    return fact.value or fact.subject


def _make_node(
    source: str,
    kind: str,
    label: str,
    start_line: int,
    end_line: int,
    quote: str,
    confidence: str,
    metadata: dict[str, object] | None = None,
) -> MarkdownNode:
    canonical = label.strip()
    evidence = MarkdownEvidence(
        source_path=source,
        start_line=start_line,
        end_line=end_line,
        quote=quote.strip(),
        confidence=confidence,
    )
    return MarkdownNode(
        id=_node_id(source, kind, canonical, start_line, end_line, evidence.quote),
        kind=kind,
        label=canonical,
        canonical_name=canonical,
        evidence=evidence,
        metadata=metadata or {},
    )


def _mention_node(
    source: str,
    kind: str,
    label: str,
    block: _Block,
    confidence: str,
    metadata: dict[str, object],
) -> MarkdownNode:
    return _make_node(source, kind, label.strip(), block.start_line, block.end_line, block.text, confidence, metadata)


def _relation(
    source: str,
    target: str,
    relation: str,
    evidence: MarkdownEvidence,
    confidence: str,
) -> MarkdownRelation:
    return MarkdownRelation(source=source, target=target, relation=relation, evidence=evidence, confidence=confidence)


def _mention_relation(parent_text: str, node: MarkdownNode) -> str:
    if node.evidence.confidence == AMBIGUOUS:
        return "SEMANTIC_CANDIDATE"
    if EXPLAINS_RE.search(parent_text):
        return "DOC_EXPLAINS"
    return "DOC_MENTIONS"


def _section_parent(section_nodes: list[tuple[_Block, MarkdownNode]], block: _Block) -> str | None:
    candidates = [
        node
        for section_block, node in section_nodes
        if section_block.start_line <= block.start_line <= node.evidence.end_line
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda node: (int(node.metadata.get("level", 0)), node.evidence.start_line), reverse=True)
    return candidates[0].id


def _extract_links(text: str) -> list[tuple[str, str]]:
    return [(match.group(1).strip(), match.group(2).strip()) for match in LINK_RE.finditer(text)]


def _backticked_values(text: str) -> Iterable[str]:
    for match in BACKTICK_RE.finditer(text):
        value = match.group(1).strip()
        if value:
            yield value


def _sentences(text: str) -> list[str]:
    normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return [sentence.strip(" -") for sentence in SENTENCE_RE.split(normalized) if sentence.strip(" -")]


def _sentence_block(block: _Block, sentence: str) -> _Block:
    return _Block("paragraph", block.start_line, block.end_line, sentence, {})


def _dedupe_nodes(nodes: list[MarkdownNode]) -> list[MarkdownNode]:
    seen: set[tuple[str, str, int, int, str]] = set()
    result: list[MarkdownNode] = []
    for node in nodes:
        key = (
            node.kind,
            node.canonical_name,
            node.evidence.start_line,
            node.evidence.end_line,
            str(node.metadata.get("reference_type", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(node)
    return result


def _block_label(block: _Block) -> str:
    if block.kind == "table":
        return "Markdown table"
    if block.kind == "code_block":
        language = block.metadata.get("language")
        return f"{language} code block" if language else "Code block"
    return _short_label(block.text)


def _short_label(text: str, limit: int = 96) -> str:
    compact = " ".join(text.strip().split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def _error_label(sentence: str) -> str:
    for value in _backticked_values(sentence):
        return _short_label(value)
    match = ERROR_RE.search(sentence)
    if match:
        for group in match.groups()[1:]:
            if group:
                return _short_label(group)
    return _short_label(sentence)


def _clean_heading(line: str) -> str:
    match = HEADING_RE.match(line)
    return match.group(2).strip() if match else line.strip()


def _fallback_title(path: Path, lines: list[str]) -> str:
    for line in lines:
        if line.strip():
            return _short_label(line)
    return path.stem


def _first_nonblank_line(lines: list[str]) -> int:
    for index, line in enumerate(lines, start=1):
        if line.strip():
            return index
    return 1


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index] and bool(TABLE_SEPARATOR_RE.match(lines[index + 1]))


def _looks_like_api(value: str) -> bool:
    return value.startswith("/api/") or bool(ENDPOINT_RE.search(value))


def _looks_like_file_path(value: str) -> bool:
    return "://" not in value and bool(FILE_PATH_RE.search(value))


def _looks_like_function(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?\(\)$", value))


def _looks_like_config_key(value: str) -> bool:
    return bool(CONFIG_KEY_RE.fullmatch(value)) and ("." in value or "_" in value or "-" in value)


def _looks_like_status(value: str) -> bool:
    if " " in value or "/" in value or "." in value:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]{2,}|[a-z]+(?:-[a-z0-9]+)+", value))


def _node_id(source: str, kind: str, label: str, start_line: int, end_line: int, quote: str) -> str:
    value = "\n".join([source, kind, label, str(start_line), str(end_line), quote])
    return f"md:{sha256_text(value)[:24]}"
