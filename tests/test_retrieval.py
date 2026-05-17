from neurograph.graph.retrieval import find_seed_nodes, infer_intent, produce_retrieval_result
from neurograph.indexer.planner import run_index
from neurograph.lifecycle import init_project


def test_retrieval_returns_code_and_document_evidence_for_api_change(tmp_path):
    init_project(tmp_path)
    (tmp_path / "README.md").write_text(
        "# API\n\nPOST /api/users creates a user and requires `email`.\n",
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

    result = produce_retrieval_result(tmp_path, "Change POST /api/users to require email", limit=6)

    assert result.intent == "impact"
    assert any("exact_endpoint" in seed.reasons for seed in result.seeds)
    labels = {(item.node_kind, item.label, item.path) for item in result.evidence}
    assert ("APIReference", "POST /api/users", "README.md") in labels
    assert ("Endpoint", "POST /api/users", "app.ts") in labels
    assert any(kind == "Function" and label == "createUser" for kind, label, _ in labels)
    assert all(len(item.quote) <= 900 for item in result.evidence)


def test_retrieval_exact_field_and_status_seeds(tmp_path):
    init_project(tmp_path)
    (tmp_path / "rules.md").write_text(
        "# Rules\n\nThe `amount_cents` field is required. Status READY means settled.\n",
        encoding="utf-8",
    )
    (tmp_path / "schema.sql").write_text(
        "CREATE TABLE payments (\n"
        "  id integer,\n"
        "  amount_cents integer,\n"
        "  status text\n"
        ");\n",
        encoding="utf-8",
    )
    (tmp_path / "states.yaml").write_text("states:\n  READY: true\n", encoding="utf-8")
    run_index(tmp_path)

    field_result = produce_retrieval_result(tmp_path, "Update amount_cents field validation", limit=6)
    status_seeds = find_seed_nodes(tmp_path, "READY status")

    assert any(item.node_kind == "Column" and item.label == "amount_cents" for item in field_result.evidence)
    assert any(seed.node.kind == "StatusValue" and seed.node.label == "READY" for seed in status_seeds)
    assert any(seed.node.kind == "ConfigKey" and seed.node.label == "READY" for seed in status_seeds)


def test_retrieval_vague_query_avoids_generic_code_matches(tmp_path):
    init_project(tmp_path)
    (tmp_path / "app.py").write_text(
        "def login():\n"
        "    return True\n\n"
        "class PaymentService:\n"
        "    pass\n",
        encoding="utf-8",
    )
    run_index(tmp_path)

    result = produce_retrieval_result(tmp_path, "user login payment flow", limit=6)

    assert infer_intent("user login payment flow") == "auto"
    assert result.seeds == ()
    assert result.evidence == ()
    assert "No exact path, symbol, endpoint, table, config key, or status seed matched the task." in result.unknowns
