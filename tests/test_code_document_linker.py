from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from neurograph import storage
from neurograph.indexer.planner import run_index
from neurograph.lifecycle import init_project


def test_links_sb_pdf_endpoint_to_code_endpoint(tmp_path):
    init_project(tmp_path)
    src = tmp_path / "src"
    docs = tmp_path / "docs"
    src.mkdir()
    docs.mkdir()
    (src / "app.ts").write_text(
        "import express from 'express';\n"
        "const app = express();\n"
        "function loginHandler(req, res) {\n"
        "  return res.json({ status: 'READY' });\n"
        "}\n"
        "app.post('/api/login', loginHandler);\n",
        encoding="utf-8",
    )
    _write_text_pdf(
        docs / "planning.pdf",
        "Login screen calls POST /api/login. Status READY means the session is ready.",
    )

    result = run_index(tmp_path)

    assert result.code_doc_links >= 1
    with storage.connect(tmp_path) as con:
        rows = con.execute(
            """
            SELECT s.kind, s.label, e.relation, t.kind, t.label, e.confidence
            FROM edges e
            JOIN nodes s ON s.id = e.src
            JOIN nodes t ON t.id = e.dst
            WHERE e.extractor = 'code_document_linker'
            ORDER BY e.relation, s.label, t.label
            """
        ).fetchall()

    assert ("APIReference", "POST /api/login", "SB_REFERENCES_API", "Endpoint", "POST /api/login", "DOC_EXACT") in rows


def test_links_markdown_api_reference_to_openapi_and_handler(tmp_path):
    init_project(tmp_path)
    (tmp_path / "README.md").write_text(
        "# API\n\nThe client calls POST /api/users to create a user.\n",
        encoding="utf-8",
    )
    (tmp_path / "openapi.yaml").write_text(
        "openapi: 3.1.0\n"
        "paths:\n"
        "  /api/users:\n"
        "    post:\n"
        "      operationId: createUser\n",
        encoding="utf-8",
    )
    (tmp_path / "app.ts").write_text(
        "function createUser(req, res) {\n"
        "  return res.json({ ok: true });\n"
        "}\n"
        "app.post('/api/users', createUser);\n",
        encoding="utf-8",
    )

    run_index(tmp_path)

    with storage.connect(tmp_path) as con:
        doc_links = con.execute(
            """
            SELECT s.kind, s.label, e.relation, t.kind, t.label, t.path
            FROM edges e
            JOIN nodes s ON s.id = e.src
            JOIN nodes t ON t.id = e.dst
            WHERE e.extractor = 'code_document_linker'
              AND s.kind = 'APIReference'
            ORDER BY t.path
            """
        ).fetchall()
        openapi_handler_links = con.execute(
            """
            SELECT s.kind, s.label, s.path, e.relation, t.kind, t.label
            FROM edges e
            JOIN nodes s ON s.id = e.src
            JOIN nodes t ON t.id = e.dst
            WHERE e.extractor = 'code_document_linker'
              AND s.kind = 'Endpoint'
              AND s.path = 'openapi.yaml'
              AND t.kind = 'Function'
            """
        ).fetchall()

    assert ("APIReference", "POST /api/users", "DOC_MENTIONS", "Endpoint", "POST /api/users", "app.ts") in doc_links
    assert ("APIReference", "POST /api/users", "DOC_MENTIONS", "Endpoint", "POST /api/users", "openapi.yaml") in doc_links
    assert ("Endpoint", "POST /api/users", "openapi.yaml", "DOC_MENTIONS", "Function", "createUser") in openapi_handler_links


def test_links_validation_rule_only_with_exact_field_and_validation_code(tmp_path):
    init_project(tmp_path)
    (tmp_path / "rules.md").write_text(
        "# Rules\n\nThe `email` field is required and cannot be empty.\n",
        encoding="utf-8",
    )
    (tmp_path / "validators.py").write_text(
        "def validate_email(email):\n"
        "    if not email:\n"
        "        raise ValueError('required')\n",
        encoding="utf-8",
    )

    run_index(tmp_path)

    with storage.connect(tmp_path) as con:
        rows = con.execute(
            """
            SELECT s.kind, s.label, e.relation, t.kind, t.label
            FROM edges e
            JOIN nodes s ON s.id = e.src
            JOIN nodes t ON t.id = e.dst
            WHERE e.extractor = 'code_document_linker'
              AND e.relation = 'SB_DEFINES_VALIDATION'
            """
        ).fetchall()

    assert ("ValidationRule", "The `email` field is required and cannot be empty.", "SB_DEFINES_VALIDATION", "Function", "validate_email") in rows


def test_avoids_false_positives_from_vague_generic_text(tmp_path):
    init_project(tmp_path)
    (tmp_path / "notes.md").write_text(
        "# Planning\n\nThe user login payment flow should be simple.\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "def login():\n"
        "    return True\n\n"
        "class PaymentService:\n"
        "    pass\n",
        encoding="utf-8",
    )

    result = run_index(tmp_path)

    assert result.code_doc_links == 0
    assert result.semantic_candidates == 0
    with storage.connect(tmp_path) as con:
        count = con.execute("SELECT COUNT(*) FROM edges WHERE extractor = 'code_document_linker'").fetchone()[0]
    assert count == 0


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
