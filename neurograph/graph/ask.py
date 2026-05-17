"""Deterministic terminal summaries for `ng ask`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neurograph.graph.context_pack import DEFAULT_TOKEN_BUDGET, ContextPack, create_context_pack
from neurograph.utils.paths import as_project_path


HIGH_CONFIDENCE = {"EXACT_COMPILER", "EXACT_STATIC", "DOC_EXACT"}
CODE_KINDS = {"Endpoint", "Function", "Method", "Class", "File", "Table", "Column", "ConfigKey", "Symbol"}
DOC_KINDS = {
    "APIReference",
    "Document",
    "Section",
    "Requirement",
    "Policy",
    "ValidationRule",
    "ErrorMessage",
    "StatusValue",
    "Screen",
    "BusinessPolicy",
    "PermissionRule",
}
MAX_ITEMS = 6
QUOTE_LIMIT = 180


def ask_project(
    root: Path,
    question: str,
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    mode: str = "auto",
) -> tuple[ContextPack, str]:
    """Build and save a Context Pack, then render a concise evidence-backed summary."""

    pack = create_context_pack(root, question, token_budget=token_budget, mode=mode, save=True)
    return pack, render_ask_summary(root, pack)


def render_ask_summary(root: Path, pack: ContextPack) -> str:
    payload = pack.payload
    lines: list[str] = []

    _section(lines, "Conclusion", [_conclusion(payload)])
    _section(lines, "High-confidence findings", _findings(payload))
    _section(lines, "Affected code", _node_lines(payload.get("affected_code", []), CODE_KINDS))
    _section(lines, "Related documents", _node_lines(payload.get("related_documents", []), DOC_KINDS))
    _section(lines, "Risks", _string_lines(payload.get("risks", [])))
    _section(lines, "Conflicts", _conflict_lines(payload.get("conflicts", [])))
    _section(lines, "Unknowns", _string_lines(payload.get("unknowns", [])))
    _section(lines, "Saved Context Pack path", [as_project_path(root, pack.path)])
    return "\n".join(lines).rstrip() + "\n"


def _conclusion(payload: dict[str, Any]) -> str:
    conflicts = payload.get("conflicts") or []
    evidence = payload.get("evidence_paths") or []
    affected_code = payload.get("affected_code") or []
    related_documents = payload.get("related_documents") or []
    mode = payload.get("mode") or "auto"
    if conflicts:
        return f"Found {len(conflicts)} concrete document-code conflict(s) in {mode} mode; inspect the listed evidence before editing."
    if affected_code and related_documents:
        return f"Found grounded code and document evidence in {mode} mode; review affected code and related documents below."
    if affected_code:
        return f"Found grounded code evidence in {mode} mode; document support is limited."
    if evidence:
        return f"Found grounded evidence in {mode} mode, but no directly affected code node was promoted."
    return f"No grounded project evidence was found in {mode} mode; treat this result as insufficient."


def _findings(payload: dict[str, Any]) -> list[str]:
    evidence = [
        item
        for item in payload.get("evidence_paths", [])
        if item.get("confidence") in HIGH_CONFIDENCE
    ]
    return [_evidence_line(item) for item in evidence[:MAX_ITEMS]]


def _node_lines(items: list[dict[str, Any]], allowed_kinds: set[str]) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        kind = str(item.get("kind") or "")
        if kind not in allowed_kinds:
            continue
        line = _node_line(item)
        key = (kind, str(item.get("label") or ""), str(item.get("location") or item.get("path") or ""))
        if line and key not in seen:
            seen.add(key)
            lines.append(line)
        if len(lines) >= MAX_ITEMS:
            break
    return lines


def _node_line(item: dict[str, Any]) -> str:
    label = item.get("label") or item.get("id") or "unnamed"
    kind = item.get("kind") or "Node"
    location = item.get("location") or item.get("path") or "unknown location"
    confidence = item.get("confidence") or "unknown"
    reasons = _reasons(item)
    evidence = _short(item.get("evidence") or "")
    suffix = f" - {evidence}" if evidence else ""
    return f"{kind} {label} at {location} [{confidence}; {reasons}]{suffix}"


def _evidence_line(item: dict[str, Any]) -> str:
    kind = item.get("kind") or "Evidence"
    label = item.get("label") or "unnamed"
    location = item.get("location") or item.get("path") or "unknown location"
    confidence = item.get("confidence") or "unknown"
    reasons = _reasons(item)
    quote = _short(item.get("quote") or "")
    suffix = f" - {quote}" if quote else ""
    return f"{kind} {label} at {location} [{confidence}; {reasons}]{suffix}"


def _conflict_lines(conflicts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for conflict in conflicts[:MAX_ITEMS]:
        summary = conflict.get("summary") or conflict.get("type") or "Conflict detected."
        confidence = conflict.get("confidence") or "unknown"
        doc = conflict.get("document_evidence") or {}
        code = conflict.get("code_evidence") or {}
        doc_location = doc.get("location") or doc.get("path") or "unknown document location"
        code_location = code.get("location") or code.get("path") or "unknown code location"
        lines.append(f"{summary} [{confidence}] doc={doc_location}; code={code_location}")
    return lines


def _string_lines(items: list[Any]) -> list[str]:
    return [_short(item, limit=260) for item in items[:MAX_ITEMS] if str(item).strip()]


def _section(lines: list[str], title: str, items: list[str]) -> None:
    if lines:
        lines.append("")
    lines.append(title)
    if items:
        lines.extend(f"- {item}" for item in items)
    else:
        lines.append("- none")


def _reasons(item: dict[str, Any]) -> str:
    reasons = item.get("reasons") or []
    if not reasons:
        return "no explicit retrieval reason"
    return "+".join(str(reason) for reason in reasons)


def _short(value: Any, *, limit: int = QUOTE_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
