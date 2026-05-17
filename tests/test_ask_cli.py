from typer.testing import CliRunner

from neurograph import storage
from neurograph.cli import app


runner = CliRunner()


def test_ng_ask_prints_summary_with_evidence_and_saves_pack(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text(
        "# API\n\nPOST /api/users creates a user and requires `email`.\n",
        encoding="utf-8",
    )
    (tmp_path / "app.ts").write_text(
        "function createUser(req, res) {\n"
        "  return res.json({ ok: true });\n"
        "}\n"
        "app.post('/api/users', createUser);\n",
        encoding="utf-8",
    )

    assert runner.invoke(app, ["init"]).exit_code == 0
    assert runner.invoke(app, ["index"]).exit_code == 0
    result = runner.invoke(app, ["ask", "Change POST /api/users to require email", "--token-budget", "1000"])

    assert result.exit_code == 0
    assert "Conclusion" in result.output
    assert "High-confidence findings" in result.output
    assert "Affected code" in result.output
    assert "Related documents" in result.output
    assert "Risks" in result.output
    assert "Conflicts" in result.output
    assert "Unknowns" in result.output
    assert "Saved Context Pack path" in result.output
    assert "POST /api/users" in result.output
    assert "README.md:3" in result.output
    assert "app.ts:4" in result.output
    assert "context_pack_version:" not in result.output
    assert list((tmp_path / ".neurograph" / "context").glob("*.yaml"))
    assert storage.get_counts(tmp_path)["context_packs"] == 1


def test_ng_ask_reports_conflicts_with_both_evidence_locations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rules.md").write_text(
        "# Rules\n\n비밀번호는 8자 이상이어야 한다.\n",
        encoding="utf-8",
    )
    (tmp_path / "validators.ts").write_text(
        "const passwordRule = minLength(6);\n",
        encoding="utf-8",
    )

    assert runner.invoke(app, ["init"]).exit_code == 0
    assert runner.invoke(app, ["index"]).exit_code == 0
    result = runner.invoke(app, ["ask", "회원가입 SB 문서와 코드가 충돌하는 부분 찾아줘"])

    assert result.exit_code == 0
    assert "Conflicts" in result.output
    assert "Document requires min length 8, but code enforces 6." in result.output
    assert "doc=rules.md:3" in result.output
    assert "code=validators.ts:1" in result.output
    assert "Saved Context Pack path" in result.output


def test_ng_ask_handles_weak_retrieval_without_inventing_details(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text(
        "def login():\n"
        "    return True\n",
        encoding="utf-8",
    )

    assert runner.invoke(app, ["init"]).exit_code == 0
    assert runner.invoke(app, ["index"]).exit_code == 0
    result = runner.invoke(app, ["ask", "로그인 retry 로직 왜 이렇게 되어 있어?"])

    assert result.exit_code == 0
    assert "Conclusion" in result.output
    assert "No grounded project evidence was found" in result.output
    assert "No exact seed matched the task" in result.output
    assert "Affected code\n- none" in result.output
    assert "Saved Context Pack path" in result.output
