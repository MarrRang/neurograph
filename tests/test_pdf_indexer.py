from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from neurograph import storage
from neurograph.graph.builder import build_file_index
from neurograph.indexer.pdf import (
    DOC_EXACT,
    DOC_INFERRED,
    EXTRACTOR,
    STATUS_TEXT_UNAVAILABLE,
    index_pdf,
)
from neurograph.utils.hashing import sha256_file


def test_pdf_fast_indexer_extracts_pages_and_sb_facts(tmp_path):
    pdf = tmp_path / "planning.pdf"
    _write_text_pdf(
        pdf,
        "Login screen lets the user click Submit. "
        "POST /api/login returns READY. "
        "Email field is required and cannot be empty. "
        "Show error message `Invalid email`. "
        "Only admins can approve refunds.",
    )

    indexed = index_pdf(pdf, "docs/planning.pdf")
    nodes_by_kind = _nodes_by_kind(indexed.nodes)

    assert indexed.status == "indexed"
    assert indexed.pages[0].page == 1
    assert nodes_by_kind["Page"][0].evidence.extractor == EXTRACTOR
    assert any(node.kind == "Screen" and "Login" in node.label for node in indexed.nodes)
    assert any(node.kind == "UserAction" and "click Submit" in node.label for node in indexed.nodes)
    assert any(node.kind == "APIReference" and node.label == "POST /api/login" for node in indexed.nodes)
    assert any(node.kind == "FormField" and node.label == "Email" for node in indexed.nodes)
    assert any(node.kind == "ValidationRule" for node in indexed.nodes)
    assert any(node.kind == "ErrorMessage" and node.label == "Invalid email" for node in indexed.nodes)
    assert any(node.kind == "StatusValue" and node.label == "READY" for node in indexed.nodes)
    assert any(node.kind == "PermissionRule" for node in indexed.nodes)
    assert all(node.evidence.page >= 1 and node.evidence.quote and node.evidence.extractor == EXTRACTOR for node in indexed.nodes)
    assert any(relation.relation == "SB_DEFINES_SCREEN" for relation in indexed.relations)
    assert any(relation.relation == "SB_DEFINES_VALIDATION" for relation in indexed.relations)
    assert any(relation.relation == "SB_REFERENCES_API" for relation in indexed.relations)
    assert any(relation.relation == "SB_REFERENCES_STATUS" for relation in indexed.relations)


def test_scanned_pdf_is_recorded_as_text_unavailable(tmp_path):
    pdf = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf.open("wb") as handle:
        writer.write(handle)

    indexed = index_pdf(pdf, "docs/scan.pdf")

    assert indexed.status == STATUS_TEXT_UNAVAILABLE
    assert indexed.pages == ()
    assert "OCR, vision, and deep parsing are off" in indexed.unknowns[0]
    assert any(node.kind == "Document" for node in indexed.nodes)
    assert any(node.kind == "Page" and node.evidence.confidence == "AMBIGUOUS" for node in indexed.nodes)


def test_pdf_nodes_are_persisted_with_page_evidence(tmp_path):
    storage.init_db(tmp_path)
    pdf = tmp_path / "planning.pdf"
    _write_text_pdf(pdf, "Settings page calls GET /api/settings. Status READY means the screen is loaded.")

    record, chunks, edges = build_file_index(tmp_path, pdf, "docs/planning.pdf", "pdf", sha256_file(pdf))
    storage.replace_file_index(tmp_path, record, chunks, edges)

    with storage.connect(tmp_path) as con:
        artifact = con.execute("SELECT kind, metadata FROM artifacts WHERE path = 'docs/planning.pdf'").fetchone()
        node_rows = con.execute("SELECT kind, label, page, confidence FROM nodes ORDER BY kind, label").fetchall()
        evidence_rows = con.execute("SELECT page, extractor, confidence, quote FROM evidence ORDER BY id").fetchall()
        relations = {row[0] for row in con.execute("SELECT relation FROM edges").fetchall()}

    assert artifact[0] == "pdf"
    assert ("Page", "Page 1", 1, DOC_EXACT) in node_rows
    assert any(row[0] == "APIReference" and row[1] == "GET /api/settings" for row in node_rows)
    assert all(row[0] == 1 and row[1] == EXTRACTOR and row[2] in {DOC_EXACT, DOC_INFERRED, "AMBIGUOUS"} and row[3] for row in evidence_rows)
    assert "CONTAINS" in relations
    assert "SB_REFERENCES_API" in relations


def _write_text_pdf(path, text):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({_escape_pdf_text(text)}) Tj ET".encode("utf-8"))
    page[NameObject("/Contents")] = stream
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    with path.open("wb") as handle:
        writer.write(handle)
    assert PdfReader(str(path)).pages[0].extract_text()


def _escape_pdf_text(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _nodes_by_kind(nodes):
    by_kind = {}
    for node in nodes:
        by_kind.setdefault(node.kind, []).append(node)
    return by_kind
