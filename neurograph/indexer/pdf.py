"""Fast page-level PDF text indexing for v0.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from neurograph.indexer.sb_facts import SBFact, extract_sb_facts, fact_id
from neurograph.utils.hashing import sha256_text


DOC_EXACT = "DOC_EXACT"
DOC_INFERRED = "DOC_INFERRED"
AMBIGUOUS = "AMBIGUOUS"
EXTRACTOR = "pdf_fast_text"

STATUS_INDEXED = "indexed"
STATUS_NEEDS_DEEP_PARSE = "needs_deep_parse"
STATUS_TEXT_UNAVAILABLE = "text_unavailable"

ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[A-Za-z0-9_./:{}?=&%+-]+)")
BARE_API_RE = re.compile(r"(?<![\w/])(/api/[A-Za-z0-9_./:{}?=&%+-]+)")
SCREEN_RE = re.compile(r"\b([A-Z][A-Za-z0-9 /_-]{1,60}?)\s+(?:screen|page|view)\b")
SCREEN_LABEL_RE = re.compile(r"\b(?:screen|page|view)\s*[:=-]\s*([A-Z][A-Za-z0-9 /_-]{1,60})", re.IGNORECASE)
ACTION_RE = re.compile(
    r"\b(?:user|admin|operator|customer|agent)\s+(?:can|must|should|may|will)?\s*"
    r"(clicks?|taps?|selects?|enters?|opens?|submits?|uploads?|downloads?|chooses?|navigates?|saves?|deletes?)\b"
    r"([^.!?\n]{0,80})",
    re.IGNORECASE,
)
FORM_FIELD_RE = re.compile(
    r"(?:field|input|form field)\s*[:=-]?\s*`?([A-Za-z][A-Za-z0-9_.-]{1,60})`?"
    r"|`([A-Za-z][A-Za-z0-9_.-]{1,60})`\s+(?:field|input)"
    r"|([A-Z][A-Za-z0-9 ]{1,40})\s+field\b"
)
VALIDATION_RE = re.compile(
    r"\b("
    r"must be valid|must match|must include|must contain|required field|required when|"
    r"cannot be empty|must not exceed|at least|at most|minimum|max(?:imum)?|"
    r"invalid if|reject(?:s|ed)? when|validation|validate"
    r")\b",
    re.IGNORECASE,
)
ERROR_RE = re.compile(
    r"\b(error(?: message)?|shows?|displays?|returns?)\b\s*(?:[:\-]\s*)?(?:`([^`]+)`|\"([^\"]+)\"|'([^']+)'|([A-Z][^.!?\n]{3,}))?",
    re.IGNORECASE,
)
STATUS_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,}|[a-z]+(?:-[a-z0-9]+)+)\b")
BUSINESS_POLICY_RE = re.compile(
    r"\b(policy|business rule|only|must not|do not|cannot|retention|approval|local-first|read-only)\b",
    re.IGNORECASE,
)
PERMISSION_RE = re.compile(
    r"\b(permission|role|admin|owner|member|viewer|editor|authorized|unauthorized|forbidden|"
    r"only\s+[A-Za-z ]+\s+can|requires?\s+[A-Za-z ]+\s+permission)\b",
    re.IGNORECASE,
)
REQUIREMENT_RE = re.compile(r"\b(must|shall|should|required|needs? to|has to|have to)\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class PdfEvidence:
    source_path: str
    page: int
    quote: str
    extractor: str
    confidence: str


@dataclass(frozen=True)
class PdfNode:
    id: str
    kind: str
    label: str
    canonical_name: str
    evidence: PdfEvidence
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PdfRelation:
    source: str
    target: str
    relation: str
    evidence: PdfEvidence
    confidence: str


@dataclass(frozen=True)
class PdfPage:
    page: int
    text: str


@dataclass(frozen=True)
class PdfDocument:
    source_path: str
    title: str
    status: str
    pages: tuple[PdfPage, ...]
    nodes: tuple[PdfNode, ...]
    relations: tuple[PdfRelation, ...]
    unknowns: tuple[str, ...] = ()


def index_pdf(path: Path, source_path: str | None = None) -> PdfDocument:
    """Extract page-level text and deterministic SB facts from a PDF."""

    source = source_path or path.as_posix()
    pages: list[PdfPage] = []
    unknowns: list[str] = []
    status = STATUS_INDEXED
    title = path.stem
    page_count = 0

    try:
        from pypdf import PdfReader
    except Exception:
        status = STATUS_NEEDS_DEEP_PARSE
        unknowns.append("PDF text extraction unavailable because pypdf could not be imported.")
        return _document(source, title, status, pages, unknowns, page_count=1)

    try:
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title)
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(PdfPage(page=index, text=text))
    except Exception as exc:
        status = STATUS_NEEDS_DEEP_PARSE
        unknowns.append(f"PDF text extraction failed: {exc}")
        return _document(source, title, status, pages, unknowns, page_count=max(page_count, 1))

    if not pages:
        status = STATUS_TEXT_UNAVAILABLE
        unknowns.append("No extractable PDF text found. OCR, vision, and deep parsing are off by default in v0.1.")

    return _document(source, title, status, pages, unknowns, page_count=max(page_count, len(pages), 1))


def parse_pdf(path: Path) -> tuple[str | None, list[PdfPage], list[str]]:
    """Compatibility wrapper for callers that only need page text."""

    document = index_pdf(path)
    return document.title, list(document.pages), list(document.unknowns)


def _document(
    source: str,
    title: str,
    status: str,
    pages: list[PdfPage],
    unknowns: list[str],
    page_count: int,
) -> PdfDocument:
    nodes: list[PdfNode] = []
    relations: list[PdfRelation] = []
    seen: set[tuple[str, str, int, str]] = set()
    first_quote = pages[0].text if pages else f"{title} ({status})"
    document_evidence = PdfEvidence(source, 1, _short_label(first_quote, 500), EXTRACTOR, DOC_EXACT if pages else AMBIGUOUS)
    document_node = _make_node(source, "Document", title, document_evidence, {"status": status, "page_count": page_count})
    nodes.append(document_node)

    pages_by_number = {page.page: page for page in pages}
    for page_number in range(1, page_count + 1):
        page = pages_by_number.get(page_number)
        page_quote = page.text if page else f"Page {page_number}: no extractable text"
        page_confidence = DOC_EXACT if page and page.text else AMBIGUOUS
        page_evidence = PdfEvidence(source, page_number, page_quote, EXTRACTOR, page_confidence)
        page_node = _make_node(source, "Page", f"Page {page_number}", page_evidence, {"status": status})
        nodes.append(page_node)
        relations.append(_relation(document_node.id, page_node.id, "CONTAINS", page_evidence, page_confidence))

        if not page or not page.text:
            continue
        for fact in _extract_page_facts(source, page):
            key = (fact.kind, fact.canonical_name, fact.evidence.page, fact.evidence.quote)
            if key not in seen:
                nodes.append(fact)
                seen.add(key)
            relations.append(_relation(page_node.id, fact.id, _relation_for_fact(fact), fact.evidence, fact.evidence.confidence))

    return PdfDocument(
        source_path=source,
        title=title,
        status=status,
        pages=tuple(pages),
        nodes=tuple(nodes),
        relations=tuple(relations),
        unknowns=tuple(unknowns),
    )


def _extract_page_facts(source: str, page: PdfPage) -> list[PdfNode]:
    return _dedupe([_node_from_fact(source, fact, page) for fact in extract_sb_facts(page.text, page=page.page)])


def _node_from_fact(source: str, fact: SBFact, page: PdfPage) -> PdfNode:
    evidence = PdfEvidence(
        source_path=source,
        page=fact.page or page.page,
        quote=fact.evidence_quote,
        extractor=EXTRACTOR,
        confidence=fact.confidence,
    )
    label = _fact_label(fact)
    return PdfNode(
        id=fact_id(source, fact),
        kind=_fact_kind(fact.type),
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


def _fact(
    source: str,
    page: PdfPage,
    kind: str,
    label: str,
    confidence: str,
    metadata: dict[str, object],
) -> PdfNode:
    evidence = PdfEvidence(
        source_path=source,
        page=page.page,
        quote=page.text.strip(),
        extractor=EXTRACTOR,
        confidence=confidence,
    )
    return _make_node(source, kind, label, evidence, metadata)


def _make_node(source: str, kind: str, label: str, evidence: PdfEvidence, metadata: dict[str, object]) -> PdfNode:
    canonical = label.strip()
    return PdfNode(
        id=_node_id(source, kind, canonical, evidence.page, evidence.quote),
        kind=kind,
        label=canonical,
        canonical_name=canonical,
        evidence=evidence,
        metadata=metadata,
    )


def _relation(source: str, target: str, relation: str, evidence: PdfEvidence, confidence: str) -> PdfRelation:
    return PdfRelation(source=source, target=target, relation=relation, evidence=evidence, confidence=confidence)


def _relation_for_fact(node: PdfNode) -> str:
    if node.kind == "Screen":
        return "SB_DEFINES_SCREEN"
    if node.kind in {"Requirement", "UserAction", "BusinessPolicy", "PermissionRule", "FormField"}:
        return "SB_DEFINES_REQUIREMENT"
    if node.kind == "ValidationRule":
        return "SB_DEFINES_VALIDATION"
    if node.kind == "APIReference":
        return "SB_REFERENCES_API"
    if node.kind == "StatusValue":
        return "SB_REFERENCES_STATUS"
    return "DOC_MENTIONS"


def _sentences(text: str) -> list[str]:
    normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return [sentence.strip(" -") for sentence in SENTENCE_RE.split(normalized) if sentence.strip(" -")]


def _error_label(sentence: str) -> str:
    for value in re.findall(r"`([^`\n]+)`", sentence):
        return _short_label(value)
    match = ERROR_RE.search(sentence)
    if match:
        for group in match.groups()[1:]:
            if group:
                return _short_label(group)
    return _short_label(sentence)


def _dedupe(nodes: list[PdfNode]) -> list[PdfNode]:
    seen: set[tuple[str, str, int, str]] = set()
    result: list[PdfNode] = []
    for node in nodes:
        key = (node.kind, node.canonical_name, node.evidence.page, node.evidence.quote)
        if key in seen:
            continue
        seen.add(key)
        result.append(node)
    return result


def _looks_like_status(value: str) -> bool:
    if value in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "PDF", "API", "HTTP"}:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]{2,}|[a-z]+(?:-[a-z0-9]+)+", value))


def _short_label(text: str, limit: int = 96) -> str:
    compact = " ".join(text.strip().split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def _clean_reference(value: str) -> str:
    return value.rstrip(".,;:)]}")


def _node_id(source: str, kind: str, label: str, page: int, quote: str) -> str:
    value = "\n".join([source, kind, label, str(page), quote])
    return f"pdf:{sha256_text(value)[:24]}"
