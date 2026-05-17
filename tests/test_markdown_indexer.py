from neurograph import storage
from neurograph.graph.builder import build_file_index
from neurograph.indexer.markdown import DOC_EXACT, DOC_INFERRED, index_markdown, parse_markdown
from neurograph.utils.hashing import sha256_file


def test_markdown_parser_splits_at_headings(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text("# NeuroGraph\n\nIntro text.\n\n## CLI\n\nUse ng.\n", encoding="utf-8")

    title, sections = parse_markdown(doc)

    assert title == "NeuroGraph"
    assert len(sections) == 2
    assert sections[0].heading == "NeuroGraph"
    assert sections[0].start_line == 1
    assert sections[1].heading == "CLI"
    assert "Use ng." in sections[1].text


def test_markdown_indexer_extracts_headings_tables_and_api_mentions(tmp_path):
    doc = tmp_path / "API.md"
    doc.write_text(
        "# API Guide\n\n"
        "## Endpoints\n\n"
        "The `CreateUser` handler calls POST /api/users and explains `/api/users/{id}`. See [docs](https://example.test/docs).\n\n"
        "| Status | Meaning |\n"
        "| --- | --- |\n"
        "| ACTIVE | User can sign in |\n",
        encoding="utf-8",
    )

    indexed = index_markdown(doc, "docs/API.md")
    nodes_by_kind = _nodes_by_kind(indexed.nodes)

    assert indexed.title == "API Guide"
    assert [node.label for node in nodes_by_kind["Section"]] == ["API Guide", "Endpoints"]
    assert nodes_by_kind["Table"][0].evidence.start_line == 7
    assert any(node.label == "POST /api/users" for node in nodes_by_kind["APIReference"])
    assert any(node.label == "/api/users/{id}" for node in nodes_by_kind["APIReference"])
    assert any(node.label == "https://example.test/docs" for node in nodes_by_kind["CodeReference"])
    assert any(node.label == "ACTIVE" for node in nodes_by_kind["StatusValue"])
    assert all(node.evidence.source_path == "docs/API.md" for node in indexed.nodes)
    assert all(node.evidence.quote for node in indexed.nodes)
    assert all(node.evidence.confidence in {DOC_EXACT, DOC_INFERRED, "AMBIGUOUS"} for node in indexed.nodes)
    assert any(relation.relation == "DOC_EXPLAINS" for relation in indexed.relations)


def test_markdown_indexer_extracts_validation_rules_and_error_messages(tmp_path):
    doc = tmp_path / "rules.markdown"
    doc.write_text(
        "# Validation\n\n"
        "The `timeout_ms` value must be at least 100 and must not exceed 10000.\n\n"
        "If the token is invalid, return error message `Invalid token`.\n\n"
        "Do not call external APIs by default.\n",
        encoding="utf-8",
    )

    indexed = index_markdown(doc, "rules.markdown")
    nodes_by_kind = _nodes_by_kind(indexed.nodes)

    assert any("timeout_ms" == node.label for node in nodes_by_kind["CodeReference"])
    assert nodes_by_kind["ValidationRule"][0].evidence.confidence == DOC_INFERRED
    assert "at least 100" in nodes_by_kind["ValidationRule"][0].evidence.quote
    assert any(node.label == "Invalid token" for node in nodes_by_kind["ErrorMessage"])
    assert any(node.kind == "Policy" and "Do not call external APIs" in node.evidence.quote for node in indexed.nodes)


def test_markdown_nodes_are_persisted_with_evidence(tmp_path):
    storage.init_db(tmp_path)
    doc = tmp_path / "README.md"
    doc.write_text(
        "# NeuroGraph\n\n"
        "Requests must include `project_id`.\n\n"
        "| Field | Required |\n"
        "| --- | --- |\n"
        "| project_id | yes |\n",
        encoding="utf-8",
    )

    record, chunks, edges = build_file_index(tmp_path, doc, "README.md", "markdown", sha256_file(doc))
    storage.replace_file_index(tmp_path, record, chunks, edges)

    with storage.connect(tmp_path) as con:
        node_rows = con.execute("SELECT kind, label, confidence FROM nodes ORDER BY kind, label").fetchall()
        evidence_rows = con.execute("SELECT quote, start_line, end_line, confidence FROM evidence ORDER BY id").fetchall()

    assert ("Document", "NeuroGraph", DOC_EXACT) in node_rows
    assert any(row[0] == "Table" and row[1] == "Markdown table" for row in node_rows)
    assert any(row[0] == "Requirement" for row in node_rows)
    assert evidence_rows
    assert all(row[0] and row[1] is not None and row[2] is not None and row[3] for row in evidence_rows)


def _nodes_by_kind(nodes):
    by_kind = {}
    for node in nodes:
        by_kind.setdefault(node.kind, []).append(node)
    return by_kind
