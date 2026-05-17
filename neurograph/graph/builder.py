"""Build storage records from parsed project files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neurograph.graph.schema import Chunk, Edge, FileRecord
from neurograph.indexer.code import index_code_graph
from neurograph.indexer.markdown import index_markdown
from neurograph.indexer.pdf import index_pdf
from neurograph.indexer.sb import parse_sb
from neurograph.utils.hashing import sha256_text


@dataclass(frozen=True)
class TextSection:
    heading: str | None
    start_line: int
    end_line: int
    text: str


def evidence_for(rel_path: str, start_line: int | None = None, end_line: int | None = None, page: int | None = None) -> str:
    if page is not None:
        return f"{rel_path}:page {page}"
    if start_line is not None and end_line is not None:
        if start_line == end_line:
            return f"{rel_path}:{start_line}"
        return f"{rel_path}:{start_line}-{end_line}"
    return rel_path


def build_file_index(
    root: Path,
    path: Path,
    rel_path: str,
    kind: str,
    digest: str,
) -> tuple[FileRecord, list[Chunk], list[Edge]]:
    stat = path.stat()
    title: str | None = None
    summary: str | None = None
    chunks: list[Chunk] = []
    edges: list[Edge] = []

    if kind == "markdown":
        document = index_markdown(path, rel_path)
        title = document.title
        for node in document.nodes:
            evidence = evidence_for(rel_path, node.evidence.start_line, node.evidence.end_line)
            chunks.append(
                Chunk(
                    id=node.id,
                    file_path=rel_path,
                    kind=kind,
                    heading=node.label if node.kind in {"Document", "Section"} else None,
                    start_line=node.evidence.start_line,
                    end_line=node.evidence.end_line,
                    text=node.evidence.quote,
                    evidence=evidence,
                    metadata={
                        "document_node": {
                            "kind": node.kind,
                            "label": node.label,
                            "canonical_name": node.canonical_name,
                            "confidence": node.evidence.confidence,
                            "metadata": node.metadata,
                        },
                        "source_path": node.evidence.source_path,
                    },
                )
            )
        edges.extend(
            Edge(source=rel.source, target=rel.target, kind=rel.relation, evidence=evidence_for(rel_path, rel.evidence.start_line, rel.evidence.end_line))
            for rel in document.relations
        )
    elif kind == "code":
        graph = index_code_graph(path, rel_path, kind)
        title = rel_path
        summary = f"code_fast_graph:{graph.language}"
        for node in graph.nodes:
            evidence = evidence_for(rel_path, node.evidence.start_line, node.evidence.end_line)
            chunks.append(
                Chunk(
                    id=node.id,
                    file_path=rel_path,
                    kind=kind,
                    symbol=node.label if node.kind in {"Class", "Function", "Method"} else None,
                    start_line=node.evidence.start_line,
                    end_line=node.evidence.end_line,
                    text=node.evidence.quote,
                    evidence=evidence,
                    metadata={
                        "document_node": {
                            "kind": node.kind,
                            "label": node.label,
                            "canonical_name": node.canonical_name,
                            "confidence": node.evidence.confidence,
                            "metadata": node.metadata,
                        },
                        "source_path": node.evidence.source_path,
                        "extractor": "code_fast_graph",
                        "language": graph.language,
                    },
                )
            )
        edges.extend(
            Edge(source=edge.source, target=edge.target, kind=edge.relation, evidence=evidence_for(rel_path, edge.evidence.start_line, edge.evidence.end_line))
            for edge in graph.edges
        )
    elif kind == "pdf":
        document = index_pdf(path, rel_path)
        title = document.title
        summary = document.status if not document.unknowns else f"{document.status}: {'; '.join(document.unknowns)}"
        for node in document.nodes:
            evidence = evidence_for(rel_path, page=node.evidence.page)
            chunks.append(
                Chunk(
                    id=node.id,
                    file_path=rel_path,
                    kind=kind,
                    page=node.evidence.page,
                    text=node.evidence.quote,
                    evidence=evidence,
                    metadata={
                        "document_node": {
                            "kind": node.kind,
                            "label": node.label,
                            "canonical_name": node.canonical_name,
                            "confidence": node.evidence.confidence,
                            "metadata": node.metadata,
                        },
                        "source_path": node.evidence.source_path,
                        "extractor": node.evidence.extractor,
                        "pdf_status": document.status,
                    },
                )
            )
        edges.extend(
            Edge(source=rel.source, target=rel.target, kind=rel.relation, evidence=evidence_for(rel_path, page=rel.evidence.page))
            for rel in document.relations
        )
    elif kind == "sb":
        title, sections = parse_sb(path)
        for index, section in enumerate(sections):
            evidence = evidence_for(rel_path, section.start_line, section.end_line)
            chunks.append(
                Chunk(
                    id=_chunk_id(rel_path, index, section.text),
                    file_path=rel_path,
                    kind=kind,
                    heading=section.heading,
                    start_line=section.start_line,
                    end_line=section.end_line,
                    text=section.text,
                    evidence=evidence,
                )
            )
    elif kind in {"openapi", "sql", "config"}:
        graph = index_code_graph(path, rel_path, kind)
        title = rel_path
        summary = f"code_fast_graph:{graph.language}"
        for node in graph.nodes:
            evidence = evidence_for(rel_path, node.evidence.start_line, node.evidence.end_line)
            chunks.append(
                Chunk(
                    id=node.id,
                    file_path=rel_path,
                    kind=kind,
                    heading=node.label if node.kind in {"File", "Module"} else None,
                    start_line=node.evidence.start_line,
                    end_line=node.evidence.end_line,
                    text=node.evidence.quote,
                    evidence=evidence,
                    metadata={
                        "document_node": {
                            "kind": node.kind,
                            "label": node.label,
                            "canonical_name": node.canonical_name,
                            "confidence": node.evidence.confidence,
                            "metadata": node.metadata,
                        },
                        "source_path": node.evidence.source_path,
                        "extractor": "code_fast_graph",
                        "language": graph.language,
                    },
                )
            )
        edges.extend(
            Edge(source=edge.source, target=edge.target, kind=edge.relation, evidence=evidence_for(rel_path, edge.evidence.start_line, edge.evidence.end_line))
            for edge in graph.edges
        )
    else:
        raise ValueError(f"Unsupported index kind: {kind}")

    for chunk in chunks:
        edges.append(Edge(source=rel_path, target=chunk.id, kind="contains", evidence=chunk.evidence))

    record = FileRecord(
        path=rel_path,
        kind=kind,
        sha256=digest,
        size_bytes=stat.st_size,
        mtime=stat.st_mtime,
        title=title,
        summary=summary,
    )
    return record, chunks, edges


def _chunk_id(rel_path: str, index: int, text: str) -> str:
    return sha256_text(f"{rel_path}\n{index}\n{text}")[:24]


def parse_plain_text(path: Path, lines_per_chunk: int = 120) -> list[TextSection]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    sections: list[TextSection] = []
    for start_index in range(0, len(lines), lines_per_chunk):
        chunk_lines = lines[start_index : start_index + lines_per_chunk]
        body = "\n".join(chunk_lines).strip()
        if not body:
            continue
        start_line = start_index + 1
        end_line = start_index + len(chunk_lines)
        sections.append(TextSection(heading=None, start_line=start_line, end_line=end_line, text=body))
    if not sections and text.strip():
        sections.append(TextSection(heading=None, start_line=1, end_line=1, text=text.strip()))
    return sections
