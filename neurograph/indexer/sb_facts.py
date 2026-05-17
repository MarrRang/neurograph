"""Conservative deterministic fact extraction for SB-style documents."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from neurograph.utils.hashing import sha256_text


DOC_EXACT = "DOC_EXACT"
DOC_INFERRED = "DOC_INFERRED"
AMBIGUOUS = "AMBIGUOUS"


HTTP_ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[A-Za-z0-9_./:{}?=&%+-]+)")
BARE_API_RE = re.compile(r"(?<![\w/])(/api/[A-Za-z0-9_./:{}?=&%+-]+)")
STATUS_RE = re.compile(r"\b(PAID|FAILED|PENDING|APPROVED|READY|ACTIVE|INACTIVE|SUCCESS|ERROR|CANCELLED|REJECTED)\b")
JSON_FIELD_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([a-z][a-z0-9]*(?:_[a-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*|status|email|password|amount|amount_cents|userId)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
QUOTED_RE = re.compile(r"`([^`\n]+)`|\"([^\"]+)\"|'([^']+)'|“([^”]+)”|‘([^’]+)’")
SCREEN_RE = re.compile(
    r"(?:화면|스크린|페이지|screen|page|view)\s*[:=-]\s*([A-Za-z0-9가-힣 _/-]{2,60})"
    r"|([A-Za-z0-9가-힣 _/-]{2,60})\s*(?:화면|스크린|페이지|screen|page|view)\b",
    re.IGNORECASE,
)
EN_ACTION_RE = re.compile(
    r"\b(?:user|admin|operator|customer|agent)\s+(?:can|must|should|may|will)?\s*"
    r"(clicks?|taps?|selects?|enters?|opens?|submits?|uploads?|downloads?|chooses?|navigates?|saves?|deletes?)\b"
    r"([^.!?\n。！？]{0,80})",
    re.IGNORECASE,
)
KO_ACTION_RE = re.compile(
    r"(사용자|관리자|운영자|고객|회원|게스트)가?\s*[^.!?\n。！？]{0,60}?"
    r"(클릭|탭|선택|입력|등록|저장|삭제|승인|거절|조회|검색|업로드|다운로드|결제|취소)(?:한다|합니다|할 수 있다|해야 한다)?"
)
FIELD_CONTEXT_RE = re.compile(
    r"(?:필드|입력값|입력 필드|파라미터|parameter|field|input|key)\s*[:=-]?\s*`?([A-Za-z][A-Za-z0-9_.-]{1,60})`?"
    r"|`([A-Za-z][A-Za-z0-9_.-]{1,60})`\s*(?:필드|입력값|파라미터|field|input|key)"
    r"|([A-Za-z][A-Za-z0-9_.-]{1,60})\s*(?:필드|입력값|파라미터|field|input|key)",
    re.IGNORECASE,
)
FIELD_HINT_RE = re.compile(r"필드|입력값|파라미터|parameter|field|input|key|값", re.IGNORECASE)
VALIDATION_EXPR_RE = re.compile(
    r"(8자\s*이상|최대\s*\d+\s*회|이메일\s*형식|required|must be|must match|must include|"
    r"cannot be empty|at least|at most|minimum|max(?:imum)?|필수|유효|형식|최소\s*\d+|"
    r"\d+\s*자\s*이상|\d+\s*자\s*이하|\d+\s*회\s*이하)",
    re.IGNORECASE,
)
ERROR_HINT_RE = re.compile(r"\b(error|warning|failure)\b|오류|에러|실패|경고", re.IGNORECASE)
REQUIREMENT_RE = re.compile(r"\b(must|shall|should|required|needs? to|has to|have to)\b|해야\s*한다|필요하다|필수", re.IGNORECASE)
BUSINESS_POLICY_RE = re.compile(
    r"\b(policy|business rule|retention|approval|refund|settlement)\b|정책|비즈니스\s*규칙|사업\s*규칙|보관|환불|정산|승인\s*정책",
    re.IGNORECASE,
)
PERMISSION_RE = re.compile(
    r"\b(permission|role|admin|owner|member|viewer|editor|authorized|unauthorized|forbidden|only\s+\w+\s+can)\b"
    r"|권한|역할|관리자만|소유자만|인증된|허용되지|금지",
    re.IGNORECASE,
)
DATABASE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]{2,})\s+(?:table|entity|collection)\b"
    r"|(?:테이블|엔티티|컬렉션)\s*[:=-]?\s*([A-Za-z][A-Za-z0-9_]{2,})"
    r"|([A-Za-z][A-Za-z0-9_]{2,})\s*(?:테이블|엔티티|컬렉션)",
    re.IGNORECASE,
)
INTEGRATION_RE = re.compile(
    r"(?:external integration|integration|webhook|연동|외부\s*연동)\s*[:=-]?\s*([A-Za-z가-힣][A-Za-z0-9가-힣 _.-]{1,50})"
    r"|([A-Za-z][A-Za-z0-9_.-]{1,40})\s*(?:API\s*)?(?:연동|webhook)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SBFact:
    type: str
    subject: str
    property: str
    value: str
    text: str
    normalized_terms: tuple[str, ...]
    evidence_quote: str
    confidence: str
    page: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def extract_sb_facts(
    text: str,
    *,
    page: int | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> list[SBFact]:
    """Extract high-confidence SB facts without linking them to code."""

    facts: list[SBFact] = []
    segments = _segments(text, start_line=start_line, end_line=end_line)
    whole_quote = _quote(text)

    for segment_text, seg_start, seg_end in segments:
        facts.extend(_api_facts(segment_text, page, seg_start, seg_end))
        facts.extend(_status_facts(segment_text, page, seg_start, seg_end))
        facts.extend(_field_facts(segment_text, page, seg_start, seg_end))
        facts.extend(_screen_facts(segment_text, page, seg_start, seg_end))
        facts.extend(_action_facts(segment_text, page, seg_start, seg_end))
        facts.extend(_validation_facts(segment_text, page, seg_start, seg_end))
        facts.extend(_error_facts(segment_text, page, seg_start, seg_end))
        facts.extend(_permission_policy_requirement_facts(segment_text, page, seg_start, seg_end))
        facts.extend(_database_facts(segment_text, page, seg_start, seg_end))
        facts.extend(_integration_facts(segment_text, page, seg_start, seg_end))

    if not facts and whole_quote:
        return []
    return _dedupe(facts)


def fact_id(source: str, fact: SBFact) -> str:
    value = "\n".join(
        [
            source,
            fact.type,
            fact.subject,
            fact.property,
            fact.value,
            str(fact.page),
            str(fact.start_line),
            str(fact.end_line),
            fact.evidence_quote,
        ]
    )
    return f"sb:{sha256_text(value)[:24]}"


def _api_facts(text: str, page: int | None, start: int | None, end: int | None) -> list[SBFact]:
    facts: list[SBFact] = []
    for method, path in HTTP_ENDPOINT_RE.findall(text):
        clean_path = _clean_reference(path)
        facts.append(
            _fact("api_reference", clean_path, "method", method.upper(), text, ("api", method.lower(), clean_path), DOC_EXACT, page, start, end)
        )
    for path in BARE_API_RE.findall(text):
        clean_path = _clean_reference(path)
        facts.append(_fact("api_reference", clean_path, "path", clean_path, text, ("api", clean_path), DOC_EXACT, page, start, end))
    return facts


def _status_facts(text: str, page: int | None, start: int | None, end: int | None) -> list[SBFact]:
    return [
        _fact("status_value", "status", "value", status, text, ("status", status.lower()), DOC_EXACT, page, start, end)
        for status in STATUS_RE.findall(text)
    ]


def _field_facts(text: str, page: int | None, start: int | None, end: int | None) -> list[SBFact]:
    facts: list[SBFact] = []
    for value in BACKTICK_RE.findall(text):
        if _looks_like_field(value):
            facts.append(_fact("form_field", value, "name", value, text, ("field", _normalize(value)), DOC_EXACT, page, start, end))
    for match in FIELD_CONTEXT_RE.finditer(text):
        value = next((group for group in match.groups() if group), "")
        if value and _looks_like_field(value):
            facts.append(_fact("form_field", value, "name", value, text, ("field", _normalize(value)), DOC_EXACT, page, start, end))
    if FIELD_HINT_RE.search(text):
        for value in JSON_FIELD_RE.findall(text):
            if _looks_like_field(value):
                facts.append(_fact("form_field", value, "name", value, text, ("field", _normalize(value)), DOC_EXACT, page, start, end))
    return facts


def _screen_facts(text: str, page: int | None, start: int | None, end: int | None) -> list[SBFact]:
    facts: list[SBFact] = []
    for match in SCREEN_RE.finditer(text):
        label = next((group for group in match.groups() if group), "").strip(" :-")
        if _valid_label(label):
            facts.append(_fact("screen", _short(label), "name", _short(label), text, ("screen", _normalize(label)), DOC_INFERRED, page, start, end))
    return facts


def _action_facts(text: str, page: int | None, start: int | None, end: int | None) -> list[SBFact]:
    facts: list[SBFact] = []
    for match in EN_ACTION_RE.finditer(text):
        action = _short(f"{match.group(1)}{match.group(2)}")
        facts.append(_fact("user_action", "user", "action", action, text, ("action", _normalize(action)), DOC_INFERRED, page, start, end))
    for match in KO_ACTION_RE.finditer(text):
        actor = match.group(1)
        action = match.group(2)
        facts.append(_fact("user_action", actor, "action", action, text, ("action", _normalize(actor), _normalize(action)), DOC_INFERRED, page, start, end))
    return facts


def _validation_facts(text: str, page: int | None, start: int | None, end: int | None) -> list[SBFact]:
    match = VALIDATION_EXPR_RE.search(text)
    if not match:
        return []
    subject = _first_field(text) or "input"
    value = match.group(1)
    return [_fact("validation_rule", subject, "rule", value, text, ("validation", _normalize(subject), _normalize(value)), DOC_INFERRED, page, start, end)]


def _error_facts(text: str, page: int | None, start: int | None, end: int | None) -> list[SBFact]:
    if not ERROR_HINT_RE.search(text):
        return []
    quoted = _quoted_values(text)
    if quoted:
        return [
            _fact("error_message", "ui_message", "text", value, text, ("error", _normalize(value)), DOC_EXACT, page, start, end)
            for value in quoted
        ]
    return [_fact("error_message", "ui_message", "text", _short(text), text, ("error",), DOC_INFERRED, page, start, end)]


def _permission_policy_requirement_facts(text: str, page: int | None, start: int | None, end: int | None) -> list[SBFact]:
    facts: list[SBFact] = []
    if PERMISSION_RE.search(text):
        facts.append(_fact("permission_rule", "permission", "rule", _short(text), text, ("permission",), DOC_INFERRED, page, start, end))
        return facts
    if BUSINESS_POLICY_RE.search(text):
        facts.append(_fact("business_policy", "business_policy", "rule", _short(text), text, ("policy",), DOC_INFERRED, page, start, end))
        return facts
    if REQUIREMENT_RE.search(text):
        facts.append(_fact("requirement", "requirement", "rule", _short(text), text, ("requirement",), DOC_INFERRED, page, start, end))
    return facts


def _database_facts(text: str, page: int | None, start: int | None, end: int | None) -> list[SBFact]:
    facts: list[SBFact] = []
    for match in DATABASE_RE.finditer(text):
        entity = next((group for group in match.groups() if group), "")
        if entity:
            facts.append(_fact("database_entity", entity, "entity", entity, text, ("database", _normalize(entity)), DOC_EXACT, page, start, end))
    return facts


def _integration_facts(text: str, page: int | None, start: int | None, end: int | None) -> list[SBFact]:
    facts: list[SBFact] = []
    for match in INTEGRATION_RE.finditer(text):
        integration = next((group for group in match.groups() if group), "").strip(" :-")
        if _valid_label(integration):
            facts.append(
                _fact("external_integration", _short(integration), "integration", _short(integration), text, ("integration", _normalize(integration)), DOC_EXACT, page, start, end)
            )
    return facts


def _fact(
    fact_type: str,
    subject: str,
    prop: str,
    value: str,
    text: str,
    terms: tuple[str, ...],
    confidence: str,
    page: int | None,
    start: int | None,
    end: int | None,
) -> SBFact:
    return SBFact(
        type=fact_type,
        subject=subject.strip(),
        property=prop,
        value=value.strip(),
        text=_short(text, 240),
        normalized_terms=tuple(term for term in terms if term),
        evidence_quote=_quote(text),
        confidence=confidence,
        page=page,
        start_line=start,
        end_line=end,
    )


def _segments(text: str, *, start_line: int | None, end_line: int | None) -> list[tuple[str, int | None, int | None]]:
    lines = text.splitlines() or [text]
    segments: list[tuple[str, int | None, int | None]] = []
    base = start_line or 1
    for offset, line in enumerate(lines):
        line_number = base + offset
        stripped = line.strip()
        if not stripped:
            continue
        for sentence in re.split(r"(?<=[.!?。！？])\s+", stripped):
            clean = sentence.strip(" -•\t")
            if clean:
                segments.append((clean, line_number, line_number))
    return segments or [(_quote(text), start_line, end_line)]


def _dedupe(facts: list[SBFact]) -> list[SBFact]:
    seen: set[tuple[str, str, str, str, int | None, int | None]] = set()
    result: list[SBFact] = []
    for fact in facts:
        key = (fact.type, fact.subject, fact.property, fact.value, fact.page, fact.start_line)
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result


def _first_field(text: str) -> str | None:
    for value in BACKTICK_RE.findall(text):
        if _looks_like_field(value):
            return value
    match = FIELD_CONTEXT_RE.search(text)
    if match:
        return next((group for group in match.groups() if group), None)
    match = JSON_FIELD_RE.search(text)
    return match.group(1) if match else None


def _quoted_values(text: str) -> list[str]:
    values: list[str] = []
    for match in QUOTED_RE.finditer(text):
        value = next((group for group in match.groups() if group), "")
        if value:
            values.append(value)
    return values


def _looks_like_field(value: str) -> bool:
    if "/" in value or " " in value or len(value) > 80:
        return False
    return bool(JSON_FIELD_RE.fullmatch(value))


def _valid_label(value: str) -> bool:
    value = value.strip()
    return 1 < len(value) <= 80 and value.lower() not in {"the", "this", "a", "an", "status"}


def _clean_reference(value: str) -> str:
    return value.rstrip(".,;:)]}")


def _quote(text: str) -> str:
    return _short(text.strip(), 500)


def _short(text: str, limit: int = 96) -> str:
    compact = " ".join(text.strip().split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def _normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_")
