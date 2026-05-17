"""Conservative document-code conflict detection for v0.1."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable

from neurograph import storage


DOC_KINDS = {"markdown", "pdf", "sb"}
CODE_KINDS = {"code", "openapi", "sql", "config"}
DOC_CONFIDENCE = {"DOC_EXACT", "DOC_INFERRED"}
STRONG_CODE_CONFIDENCE = {"EXACT_STATIC", "EXACT_COMPILER"}
HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
FIELD_ALIASES = {
    "password": {"password", "passwd", "pwd", "비밀번호", "암호"},
    "email": {"email", "이메일", "email_address", "emailaddress"},
}
ROLE_ALIASES = {
    "admin": {"admin", "admins", "administrator", "관리자"},
    "owner": {"owner", "owners", "소유자"},
    "user": {"user", "users", "member", "members", "사용자", "회원"},
}
SIGNUP_SYNONYMS = {"signup", "sign-up", "register", "registration", "가입", "회원가입"}


@dataclass(frozen=True)
class EvidenceRef:
    path: str
    location: str
    quote: str
    confidence: str | None = None


@dataclass(frozen=True)
class DetectedConflict:
    type: str
    subject: str
    summary: str
    confidence: str
    document_evidence: EvidenceRef
    code_evidence: EvidenceRef
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConflictReport:
    conflicts: tuple[DetectedConflict, ...]
    risks: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()


@dataclass(frozen=True)
class _DocNode:
    kind: str
    label: str
    path: str
    start_line: int | None
    end_line: int | None
    page: int | None
    confidence: str | None
    metadata: dict[str, Any]
    text: str


@dataclass(frozen=True)
class _CodeArtifact:
    path: str
    kind: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _ValueFact:
    subject: str
    value: Any
    evidence: EvidenceRef


@dataclass(frozen=True)
class _EndpointFact:
    method: str
    path: str
    evidence: EvidenceRef


def detect_conflicts(root: Path) -> ConflictReport:
    """Return only concrete document-code contradictions with both-side evidence."""

    doc_nodes = _load_doc_nodes(root)
    code_artifacts = _load_code_artifacts(root)
    code_endpoints = _load_code_endpoints(root)
    conflicts: list[DetectedConflict] = []
    risks: list[str] = []
    unknowns: list[str] = []

    conflicts.extend(_validation_mismatches(doc_nodes, code_artifacts))
    conflicts.extend(_status_value_mismatches(doc_nodes, code_artifacts))
    conflicts.extend(_endpoint_mismatches(doc_nodes, code_endpoints))
    conflicts.extend(_retry_policy_mismatches(doc_nodes, code_artifacts))
    conflicts.extend(_timeout_mismatches(doc_nodes, code_artifacts))
    conflicts.extend(_permission_mismatches(doc_nodes, code_artifacts))

    if not doc_nodes:
        unknowns.append("No document facts were available for conflict detection.")
    if not code_artifacts and not code_endpoints:
        unknowns.append("No code facts were available for conflict detection.")
    if not conflicts:
        unknowns.append("No concrete document-code contradictions were detected by the v0.1 conflict detector.")

    return ConflictReport(
        conflicts=tuple(_dedupe_conflicts(conflicts)),
        risks=tuple(dict.fromkeys(risks)),
        unknowns=tuple(dict.fromkeys(unknowns)),
    )


def conflict_to_dict(conflict: DetectedConflict) -> dict[str, Any]:
    return {
        "type": conflict.type,
        "subject": conflict.subject,
        "summary": conflict.summary,
        "confidence": conflict.confidence,
        "document_evidence": _evidence_dict(conflict.document_evidence),
        "code_evidence": _evidence_dict(conflict.code_evidence),
        "details": conflict.details,
    }


def _load_doc_nodes(root: Path) -> list[_DocNode]:
    if not _db_exists(root):
        return []
    with storage.connect(root) as con:
        rows = con.execute(
            """
            SELECT n.kind, n.label, n.path, n.start_line, n.end_line, n.page, n.confidence, n.metadata, c.text
            FROM nodes n
            JOIN artifacts a ON a.id = n.artifact_id
            LEFT JOIN chunks c ON c.id = n.id
            WHERE a.kind IN ('markdown', 'pdf', 'sb')
            ORDER BY a.path, n.start_line, n.page, n.id
            """
        ).fetchall()
    return [
        _DocNode(
            kind=str(row[0] or ""),
            label=str(row[1] or ""),
            path=str(row[2] or ""),
            start_line=row[3],
            end_line=row[4],
            page=row[5],
            confidence=row[6],
            metadata=_json_load(row[7]),
            text=str(row[8] or row[1] or ""),
        )
        for row in rows
    ]


def _load_code_artifacts(root: Path) -> list[_CodeArtifact]:
    if not _db_exists(root):
        return []
    with storage.connect(root) as con:
        rows = con.execute(
            "SELECT path, kind FROM artifacts WHERE kind IN ('code', 'openapi', 'sql', 'config') ORDER BY path"
        ).fetchall()
    artifacts: list[_CodeArtifact] = []
    for path, kind in rows:
        source = _safe_project_file(root, str(path or ""))
        if source is None:
            continue
        if not source.exists() or not source.is_file():
            continue
        artifacts.append(
            _CodeArtifact(
                path=str(path),
                kind=str(kind),
                lines=tuple(source.read_text(encoding="utf-8", errors="replace").splitlines()),
            )
        )
    return artifacts


def _load_code_endpoints(root: Path) -> list[_EndpointFact]:
    if not _db_exists(root):
        return []
    with storage.connect(root) as con:
        rows = con.execute(
            """
            SELECT n.label, n.path, n.start_line, n.end_line, n.page, n.confidence, n.metadata, c.text
            FROM nodes n
            JOIN artifacts a ON a.id = n.artifact_id
            LEFT JOIN chunks c ON c.id = n.id
            WHERE a.kind IN ('code', 'openapi', 'config') AND n.kind = 'Endpoint'
            ORDER BY n.path, n.start_line, n.id
            """
        ).fetchall()
    endpoints: list[_EndpointFact] = []
    for label, path, start, end, page, confidence, metadata, text in rows:
        method, endpoint = _endpoint_parts(str(label or ""), _json_load(metadata))
        if method and endpoint:
            endpoints.append(
                _EndpointFact(
                    method=method,
                    path=endpoint,
                    evidence=EvidenceRef(str(path or ""), _location(str(path or ""), start, end, page), _short_quote(str(text or label or "")), confidence),
                )
            )
    return endpoints


def _validation_mismatches(doc_nodes: list[_DocNode], code_artifacts: list[_CodeArtifact]) -> list[DetectedConflict]:
    doc_facts = _doc_min_length_facts(doc_nodes)
    code_facts = _code_min_length_facts(code_artifacts)
    conflicts: list[DetectedConflict] = []
    for doc in doc_facts:
        for code in code_facts:
            if doc.subject == code.subject and doc.value != code.value:
                confidence = "high" if doc.evidence.confidence == "DOC_EXACT" else "medium"
                conflicts.append(
                    DetectedConflict(
                        type="validation_mismatch",
                        subject=doc.subject,
                        summary=f"Document requires min length {doc.value}, but code enforces {code.value}.",
                        confidence=confidence,
                        document_evidence=doc.evidence,
                        code_evidence=code.evidence,
                        details={"document_min_length": doc.value, "code_min_length": code.value},
                    )
                )
    return conflicts


def _status_value_mismatches(doc_nodes: list[_DocNode], code_artifacts: list[_CodeArtifact]) -> list[DetectedConflict]:
    doc_statuses = _doc_status_facts(doc_nodes)
    code_enums = _code_status_enums(code_artifacts)
    conflicts: list[DetectedConflict] = []
    if not doc_statuses or not code_enums:
        return conflicts
    for doc in doc_statuses:
        for enum in code_enums:
            values = set(enum.value)
            if doc.value not in values:
                conflicts.append(
                    DetectedConflict(
                        type="status_value_mismatch",
                        subject="status",
                        summary=f"Document references status {doc.value}, but code enum contains {', '.join(sorted(values))}.",
                        confidence="high" if doc.evidence.confidence == "DOC_EXACT" else "medium",
                        document_evidence=doc.evidence,
                        code_evidence=enum.evidence,
                        details={"document_status": doc.value, "code_status_values": sorted(values)},
                    )
                )
    return conflicts


def _endpoint_mismatches(doc_nodes: list[_DocNode], code_endpoints: list[_EndpointFact]) -> list[DetectedConflict]:
    doc_endpoints = _doc_endpoint_facts(doc_nodes)
    exact = {(endpoint.method, endpoint.path) for endpoint in code_endpoints}
    conflicts: list[DetectedConflict] = []
    for doc in doc_endpoints:
        if (doc.method, doc.path) in exact:
            continue
        candidate = _similar_endpoint(doc, code_endpoints)
        if candidate is None:
            continue
        conflicts.append(
            DetectedConflict(
                type="endpoint_mismatch",
                subject=f"{doc.method} {doc.path}",
                summary=f"Document names {doc.method} {doc.path}, but similar code endpoint is {candidate.method} {candidate.path}.",
                confidence="medium",
                document_evidence=doc.evidence,
                code_evidence=candidate.evidence,
                details={"document_endpoint": f"{doc.method} {doc.path}", "code_endpoint": f"{candidate.method} {candidate.path}"},
            )
        )
    return conflicts


def _retry_policy_mismatches(doc_nodes: list[_DocNode], code_artifacts: list[_CodeArtifact]) -> list[DetectedConflict]:
    return _numeric_policy_mismatches(
        "retry_policy_mismatch",
        "retry_count",
        _doc_retry_facts(doc_nodes),
        _code_retry_facts(code_artifacts),
        "Document retry count is {doc}, but code uses {code}.",
    )


def _timeout_mismatches(doc_nodes: list[_DocNode], code_artifacts: list[_CodeArtifact]) -> list[DetectedConflict]:
    return _numeric_policy_mismatches(
        "timeout_mismatch",
        "timeout_ms",
        _doc_timeout_facts(doc_nodes),
        _code_timeout_facts(code_artifacts),
        "Document timeout is {doc}ms, but code uses {code}ms.",
    )


def _permission_mismatches(doc_nodes: list[_DocNode], code_artifacts: list[_CodeArtifact]) -> list[DetectedConflict]:
    docs = _doc_permission_facts(doc_nodes)
    codes = _code_permission_facts(code_artifacts)
    conflicts: list[DetectedConflict] = []
    for doc in docs:
        for code in codes:
            code_roles = set(code.value)
            if doc.value not in code_roles:
                conflicts.append(
                    DetectedConflict(
                        type="permission_mismatch",
                        subject="permission",
                        summary=f"Document requires role {doc.value}, but code allows {', '.join(sorted(code_roles))}.",
                        confidence="medium",
                        document_evidence=doc.evidence,
                        code_evidence=code.evidence,
                        details={"document_role": doc.value, "code_roles": sorted(code_roles)},
                    )
                )
    return conflicts


def _numeric_policy_mismatches(
    conflict_type: str,
    subject: str,
    docs: list[_ValueFact],
    codes: list[_ValueFact],
    summary_template: str,
) -> list[DetectedConflict]:
    conflicts: list[DetectedConflict] = []
    for doc in docs:
        for code in codes:
            if doc.value != code.value:
                conflicts.append(
                    DetectedConflict(
                        type=conflict_type,
                        subject=subject,
                        summary=summary_template.format(doc=doc.value, code=code.value),
                        confidence="medium",
                        document_evidence=doc.evidence,
                        code_evidence=code.evidence,
                        details={"document_value": doc.value, "code_value": code.value},
                    )
                )
    return conflicts


def _doc_min_length_facts(doc_nodes: list[_DocNode]) -> list[_ValueFact]:
    facts: list[_ValueFact] = []
    for node in doc_nodes:
        if node.kind not in {"ValidationRule", "Requirement", "Paragraph"} or node.confidence not in DOC_CONFIDENCE:
            continue
        text = _doc_text(node)
        field = _field_from_text(text) or _canonical_field(str(node.metadata.get("subject") or ""))
        value = _min_length_from_doc_text(text)
        if field and value is not None:
            facts.append(_ValueFact(field, value, _doc_evidence(node)))
    return _dedupe_facts(facts)


def _code_min_length_facts(code_artifacts: list[_CodeArtifact]) -> list[_ValueFact]:
    facts: list[_ValueFact] = []
    for artifact in code_artifacts:
        for index, line in enumerate(artifact.lines):
            value = _min_length_from_code_line(line)
            if value is None:
                continue
            context = _nearby_text(artifact.lines, index, radius=3)
            field = _field_from_text(context)
            if not field:
                continue
            facts.append(
                _ValueFact(
                    field,
                    value,
                    EvidenceRef(
                        artifact.path,
                        _location(artifact.path, index + 1, index + 1, None),
                        line.strip(),
                        "EXACT_STATIC",
                    ),
                )
            )
    return _dedupe_facts(facts)


def _doc_status_facts(doc_nodes: list[_DocNode]) -> list[_ValueFact]:
    facts: list[_ValueFact] = []
    for node in doc_nodes:
        if node.kind == "StatusValue" and node.confidence == "DOC_EXACT":
            value = node.label.strip().upper()
            if _looks_like_status(value):
                facts.append(_ValueFact("status", value, _doc_evidence(node)))
    return _dedupe_facts(facts)


def _code_status_enums(code_artifacts: list[_CodeArtifact]) -> list[_ValueFact]:
    facts: list[_ValueFact] = []
    for artifact in code_artifacts:
        text = "\n".join(artifact.lines)
        for match in re.finditer(r'"status"[\s\S]{0,300}?"enum"\s*:\s*\[([^\]]+)\]', text, re.IGNORECASE):
            values = _enum_values(match.group(1))
            if values:
                line = _line_for_offset(text, match.start())
                facts.append(_ValueFact("status", tuple(values), EvidenceRef(artifact.path, _location(artifact.path, line, line, None), _short_quote(match.group(0)), "EXACT_STATIC")))
        for index, line in enumerate(artifact.lines):
            if "enum" not in line.lower():
                continue
            context = _nearby_text(artifact.lines, index, radius=6)
            if "status" not in context.lower():
                continue
            values = _enum_values(line)
            if not values:
                values = _yaml_enum_values(artifact.lines, index)
            if values:
                facts.append(_ValueFact("status", tuple(values), EvidenceRef(artifact.path, _location(artifact.path, index + 1, index + 1, None), _short_quote(context), "EXACT_STATIC")))
    return _dedupe_facts(facts)


def _doc_endpoint_facts(doc_nodes: list[_DocNode]) -> list[_EndpointFact]:
    facts: list[_EndpointFact] = []
    for node in doc_nodes:
        if node.kind != "APIReference" or node.confidence not in DOC_CONFIDENCE:
            continue
        method, endpoint = _endpoint_parts(node.label, node.metadata)
        if method and endpoint:
            facts.append(_EndpointFact(method, endpoint, _doc_evidence(node)))
    return _dedupe_endpoint_facts(facts)


def _doc_retry_facts(doc_nodes: list[_DocNode]) -> list[_ValueFact]:
    facts: list[_ValueFact] = []
    for node in doc_nodes:
        if node.confidence not in DOC_CONFIDENCE:
            continue
        text = _doc_text(node)
        match = re.search(r"\b(?:retry|retries|max retries|maxRetries)\D{0,20}(\d+)\b|최대\s*(\d+)\s*회", text, re.IGNORECASE)
        if match and ("retry" in text.lower() or "재시" in text or "최대" in text):
            value = int(next(group for group in match.groups() if group))
            facts.append(_ValueFact("retry_count", value, _doc_evidence(node)))
    return _dedupe_facts(facts)


def _code_retry_facts(code_artifacts: list[_CodeArtifact]) -> list[_ValueFact]:
    facts: list[_ValueFact] = []
    pattern = re.compile(r"\b(?:maxRetries|retries|retryCount|max_retries|retry_count)\b\s*[:=]\s*(\d+)", re.IGNORECASE)
    for artifact in code_artifacts:
        for index, line in enumerate(artifact.lines):
            match = pattern.search(line)
            if match:
                facts.append(_ValueFact("retry_count", int(match.group(1)), EvidenceRef(artifact.path, _location(artifact.path, index + 1, index + 1, None), line.strip(), "EXACT_STATIC")))
    return _dedupe_facts(facts)


def _doc_timeout_facts(doc_nodes: list[_DocNode]) -> list[_ValueFact]:
    facts: list[_ValueFact] = []
    for node in doc_nodes:
        if node.confidence not in DOC_CONFIDENCE:
            continue
        text = _doc_text(node)
        if "timeout" not in text.lower() and "타임아웃" not in text and "제한 시간" not in text:
            continue
        value = _timeout_ms(text)
        if value is not None:
            facts.append(_ValueFact("timeout_ms", value, _doc_evidence(node)))
    return _dedupe_facts(facts)


def _code_timeout_facts(code_artifacts: list[_CodeArtifact]) -> list[_ValueFact]:
    facts: list[_ValueFact] = []
    for artifact in code_artifacts:
        for index, line in enumerate(artifact.lines):
            if not re.search(r"timeout|TIMEOUT|timeoutMs|timeout_ms", line):
                continue
            value = _timeout_ms(line)
            if value is not None:
                facts.append(_ValueFact("timeout_ms", value, EvidenceRef(artifact.path, _location(artifact.path, index + 1, index + 1, None), line.strip(), "EXACT_STATIC")))
    return _dedupe_facts(facts)


def _doc_permission_facts(doc_nodes: list[_DocNode]) -> list[_ValueFact]:
    facts: list[_ValueFact] = []
    for node in doc_nodes:
        if node.kind not in {"PermissionRule", "Policy", "Requirement"} or node.confidence not in DOC_CONFIDENCE:
            continue
        text = _doc_text(node)
        if not re.search(r"\bonly\b|만\b", text, re.IGNORECASE):
            continue
        role = _role_from_text(text)
        if role:
            facts.append(_ValueFact("permission", role, _doc_evidence(node)))
    return _dedupe_facts(facts)


def _code_permission_facts(code_artifacts: list[_CodeArtifact]) -> list[_ValueFact]:
    facts: list[_ValueFact] = []
    pattern = re.compile(r"\b(?:roles|allowedRoles|requiredRoles|permissions)\b\s*[:=]\s*\[([^\]]+)\]", re.IGNORECASE)
    for artifact in code_artifacts:
        for index, line in enumerate(artifact.lines):
            match = pattern.search(line)
            if not match:
                continue
            roles = sorted({_canonical_role(value) for value in re.findall(r"[A-Za-z가-힣_]+", match.group(1)) if _canonical_role(value)})
            if roles:
                facts.append(_ValueFact("permission", tuple(roles), EvidenceRef(artifact.path, _location(artifact.path, index + 1, index + 1, None), line.strip(), "EXACT_STATIC")))
    return _dedupe_facts(facts)


def _doc_text(node: _DocNode) -> str:
    return node.text or node.label


def _doc_evidence(node: _DocNode) -> EvidenceRef:
    return EvidenceRef(
        path=node.path,
        location=_location(node.path, node.start_line, node.end_line, node.page),
        quote=_short_quote(_doc_text(node)),
        confidence=node.confidence,
    )


def _min_length_from_doc_text(text: str) -> int | None:
    patterns = [
        r"(\d+)\s*자\s*이상",
        r"(?:at least|minimum|min(?:imum)? length)\D{0,20}(\d+)",
        r"(\d+)\s*(?:characters?|chars?)\s*(?:or more|min(?:imum)?)",
        r"must be\D{0,20}(\d+)\D{0,20}(?:characters?|chars?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _min_length_from_code_line(line: str) -> int | None:
    patterns = [
        r"\bminLength\s*\(\s*(\d+)\s*\)",
        r"\bminLength\b\s*[:=]\s*(\d+)",
        r"\bmin_length\b\s*[:=]\s*(\d+)",
        r"\.min\s*\(\s*(\d+)\s*\)",
        r"\.length\s*<\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return int(match.group(1))
    return None


def _field_from_text(text: str) -> str:
    lowered = text.lower()
    for canonical, aliases in FIELD_ALIASES.items():
        if any(alias.lower() in lowered for alias in aliases):
            return canonical
    match = re.search(r"`([A-Za-z_$][\w$.-]*)`", text)
    if match:
        return _canonical_field(match.group(1))
    match = re.search(r"\b([A-Za-z_$][\w$.-]*)\s*:\s*(?:z\.string|.*minLength|.*\.min\()", text)
    if match:
        return _canonical_field(match.group(1))
    return ""


def _canonical_field(value: str) -> str:
    normalized = value.strip().strip("`'\"").lower()
    for canonical, aliases in FIELD_ALIASES.items():
        if normalized in {alias.lower() for alias in aliases}:
            return canonical
    return normalized if re.fullmatch(r"[a-z_][a-z0-9_.-]*", normalized) else ""


def _endpoint_parts(label: str, metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    method = metadata.get("method")
    path = metadata.get("path") or metadata.get("endpoint")
    method_text = method.upper() if isinstance(method, str) else None
    if isinstance(path, str) and path:
        if method_text is None:
            match = re.search(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[A-Za-z0-9_./:{}?=&%+-]+)", label, re.IGNORECASE)
            if match:
                method_text = match.group(1).upper()
        return method_text if method_text in HTTP_METHODS else None, _clean_endpoint(path)
    match = re.search(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[A-Za-z0-9_./:{}?=&%+-]+)", label, re.IGNORECASE)
    if match:
        return match.group(1).upper(), _clean_endpoint(match.group(2))
    return None, None


def _clean_endpoint(path: str) -> str:
    return path.strip().rstrip(".,;:)]}").lower()


def _similar_endpoint(doc: _EndpointFact, code_endpoints: list[_EndpointFact]) -> _EndpointFact | None:
    best: tuple[int, _EndpointFact] | None = None
    doc_terms = _endpoint_terms(doc.path)
    for endpoint in code_endpoints:
        if endpoint.method != doc.method:
            continue
        code_terms = _endpoint_terms(endpoint.path)
        overlap = len(doc_terms & code_terms)
        synonym = bool((doc_terms & SIGNUP_SYNONYMS) and (code_terms & SIGNUP_SYNONYMS))
        score = overlap + (2 if synonym else 0)
        if score >= 2 or (score >= 1 and synonym):
            if best is None or score > best[0]:
                best = (score, endpoint)
    return best[1] if best else None


def _endpoint_terms(path: str) -> set[str]:
    terms = {part for part in re.split(r"[^a-z0-9가-힣]+", path.lower()) if part}
    return terms - {"api", "v1", "v2", "v3"}


def _enum_values(text: str) -> tuple[str, ...]:
    values = [value.upper() for value in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text)]
    return tuple(dict.fromkeys(value for value in values if _looks_like_status(value)))


def _yaml_enum_values(lines: tuple[str, ...], enum_index: int) -> tuple[str, ...]:
    values: list[str] = []
    for line in lines[enum_index + 1 : enum_index + 8]:
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"-\s*['\"]?([A-Z][A-Z0-9_]{2,})['\"]?", stripped)
        if match:
            values.append(match.group(1).upper())
        elif values:
            break
    return tuple(dict.fromkeys(value for value in values if _looks_like_status(value)))


def _looks_like_status(value: str) -> bool:
    return value not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "API", "HTTP", "JSON", "PDF"}


def _timeout_ms(text: str) -> int | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(ms|milliseconds?|초|s|sec|seconds?)\b", text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        return int(value * 1000) if unit in {"s", "sec", "second", "seconds", "초"} else int(value)
    match = re.search(r"\b(?:timeout|TIMEOUT|timeoutMs|timeout_ms)\b\D{0,20}(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _role_from_text(text: str) -> str:
    lowered = text.lower()
    for role, aliases in ROLE_ALIASES.items():
        if any(alias.lower() in lowered for alias in aliases):
            return role
    return ""


def _canonical_role(value: str) -> str:
    lowered = value.lower()
    for role, aliases in ROLE_ALIASES.items():
        if lowered in {alias.lower() for alias in aliases}:
            return role
    return ""


def _nearby_text(lines: tuple[str, ...], index: int, radius: int) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(lines[start:end])


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _location(path: str, start_line: int | None, end_line: int | None, page: int | None) -> str:
    if page is not None:
        return f"{path}:page {page}"
    if start_line is not None and end_line is not None:
        if start_line == end_line:
            return f"{path}:{start_line}"
        return f"{path}:{start_line}-{end_line}"
    return path


def _short_quote(value: str, limit: int = 400) -> str:
    compact = "\n".join(line.rstrip() for line in str(value or "").strip().splitlines()).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _evidence_dict(evidence: EvidenceRef) -> dict[str, Any]:
    return {
        "path": evidence.path,
        "location": evidence.location,
        "quote": evidence.quote,
        "confidence": evidence.confidence,
    }


def _dedupe_facts(facts: Iterable[_ValueFact]) -> list[_ValueFact]:
    seen: set[tuple[str, Any, str]] = set()
    result: list[_ValueFact] = []
    for fact in facts:
        key = (fact.subject, fact.value, fact.evidence.location)
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result


def _dedupe_endpoint_facts(facts: Iterable[_EndpointFact]) -> list[_EndpointFact]:
    seen: set[tuple[str, str, str]] = set()
    result: list[_EndpointFact] = []
    for fact in facts:
        key = (fact.method, fact.path, fact.evidence.location)
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result


def _dedupe_conflicts(conflicts: Iterable[DetectedConflict]) -> list[DetectedConflict]:
    by_claim: dict[tuple[str, str, str, str, str], DetectedConflict] = {}
    for conflict in conflicts:
        key = (
            conflict.type,
            conflict.subject,
            conflict.summary,
            conflict.code_evidence.location,
            json.dumps(conflict.details, sort_keys=True),
        )
        current = by_claim.get(key)
        if current is None or _conflict_rank(conflict) < _conflict_rank(current):
            by_claim[key] = conflict
    return sorted(
        by_claim.values(),
        key=lambda item: (item.type, item.subject, item.document_evidence.location, item.code_evidence.location),
    )


def _conflict_rank(conflict: DetectedConflict) -> tuple[int, int, str]:
    confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(conflict.confidence, 3)
    return (confidence_rank, len(conflict.document_evidence.quote), conflict.document_evidence.location)


def _json_load(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _db_exists(root: Path) -> bool:
    from neurograph.utils.paths import db_path

    return db_path(root).exists()


def _safe_project_file(root: Path, path: str) -> Path | None:
    candidate = root / path
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return resolved
