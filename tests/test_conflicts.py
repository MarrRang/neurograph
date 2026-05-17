from neurograph.graph.conflicts import detect_conflicts
from neurograph.graph.context_pack import build_context_pack
from neurograph.indexer.planner import run_index
from neurograph.lifecycle import init_project


def test_detects_simple_validation_mismatch_and_context_pack_includes_it(tmp_path):
    init_project(tmp_path)
    (tmp_path / "rules.md").write_text(
        "# Rules\n\n비밀번호는 8자 이상이어야 한다.\n",
        encoding="utf-8",
    )
    (tmp_path / "validators.ts").write_text(
        "const passwordRule = minLength(6);\n",
        encoding="utf-8",
    )
    run_index(tmp_path)

    report = detect_conflicts(tmp_path)
    pack = build_context_pack(tmp_path, "비밀번호 validation", token_budget=1200)

    assert len(report.conflicts) == 1
    conflict = report.conflicts[0]
    assert conflict.type == "validation_mismatch"
    assert conflict.subject == "password"
    assert conflict.document_evidence.location == "rules.md:3"
    assert conflict.code_evidence.location == "validators.ts:1"
    assert conflict.details == {"document_min_length": 8, "code_min_length": 6}
    assert "conflicts:" in pack
    assert "validation_mismatch" in pack
    assert "Document requires min length 8, but code enforces 6." in pack


def test_dedupes_repeated_validation_mismatch_claims(tmp_path):
    init_project(tmp_path)
    (tmp_path / "signup.md").write_text(
        "# Signup SB\n\n"
        "POST /users/signup 호출.\n"
        "비밀번호는 8자 이상이어야 한다.\n"
        '실패 시 오류 메시지: "비밀번호는 8자 이상이어야 합니다"\n',
        encoding="utf-8",
    )
    (tmp_path / "server.ts").write_text(
        "export function signupHandler(req, res) {\n"
        "  if (String(req.body.password || '').length < 6) {\n"
        "    return res.status(400).json({ error: 'password too short' });\n"
        "  }\n"
        "}\n"
        "app.post('/users/signup', signupHandler);\n",
        encoding="utf-8",
    )
    run_index(tmp_path)

    report = detect_conflicts(tmp_path)

    validation_conflicts = [conflict for conflict in report.conflicts if conflict.type == "validation_mismatch"]
    assert len(validation_conflicts) == 1
    assert validation_conflicts[0].document_evidence.location == "signup.md:4"
    assert validation_conflicts[0].code_evidence.location == "server.ts:2"


def test_detects_concrete_endpoint_mismatch_with_medium_confidence(tmp_path):
    init_project(tmp_path)
    (tmp_path / "signup.md").write_text(
        "# Signup\n\nThe app calls POST /users/signup for signup.\n",
        encoding="utf-8",
    )
    (tmp_path / "app.ts").write_text(
        "function registerUser(req, res) {\n"
        "  return res.json({ ok: true });\n"
        "}\n"
        "app.post('/api/users/register', registerUser);\n",
        encoding="utf-8",
    )
    run_index(tmp_path)

    report = detect_conflicts(tmp_path)

    assert any(
        conflict.type == "endpoint_mismatch"
        and conflict.confidence == "medium"
        and conflict.details["document_endpoint"] == "POST /users/signup"
        and conflict.details["code_endpoint"] == "POST /api/users/register"
        for conflict in report.conflicts
    )


def test_detects_status_value_mismatch_only_with_exact_enum_evidence(tmp_path):
    init_project(tmp_path)
    (tmp_path / "payments.md").write_text(
        "# Payments\n\nThe payment status = PAID after capture.\n",
        encoding="utf-8",
    )
    (tmp_path / "openapi.yaml").write_text(
        "openapi: 3.1.0\n"
        "components:\n"
        "  schemas:\n"
        "    Payment:\n"
        "      properties:\n"
        "        status:\n"
        "          enum:\n"
        "            - CAPTURED\n"
        "            - FAILED\n",
        encoding="utf-8",
    )
    run_index(tmp_path)

    report = detect_conflicts(tmp_path)

    assert any(
        conflict.type == "status_value_mismatch"
        and conflict.details["document_status"] == "PAID"
        and conflict.details["code_status_values"] == ["CAPTURED", "FAILED"]
        for conflict in report.conflicts
    )


def test_avoids_vague_false_positive_conflicts(tmp_path):
    init_project(tmp_path)
    (tmp_path / "notes.md").write_text(
        "# Notes\n\nThe login payment flow should feel simple.\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "def login():\n"
        "    return True\n\n"
        "class PaymentService:\n"
        "    pass\n",
        encoding="utf-8",
    )
    run_index(tmp_path)

    report = detect_conflicts(tmp_path)
    pack = build_context_pack(tmp_path, "login payment flow", token_budget=900)

    assert report.conflicts == ()
    assert "conflicts: []" in pack
    assert "validation_mismatch" not in pack
    assert "endpoint_mismatch" not in pack
