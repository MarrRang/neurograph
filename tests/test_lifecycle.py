from typer.testing import CliRunner

from neurograph.cli import app
from neurograph.lifecycle import load_manifest


runner = CliRunner()


def test_init_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    first = runner.invoke(app, ["init"])
    second = runner.invoke(app, ["init"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert (tmp_path / ".neurograph").is_dir()
    assert (tmp_path / ".neurograph" / "cache").is_dir()
    assert (tmp_path / ".neurograph" / "context").is_dir()
    assert (tmp_path / ".neurograph" / "install-manifest.json").is_file()
    assert (tmp_path / ".neurograph" / "brain.duckdb").is_file()
    assert (tmp_path / ".neurographignore").is_file()
    assert "Already initialized." in second.output
    manifest = load_manifest(tmp_path)
    assert manifest is not None
    assert ".neurograph/brain.duckdb" in manifest["created_files"]
    assert ".neurograph/install-manifest.json" in manifest["created_files"]
    assert ".neurograph" in manifest["created_directories"]
    assert "created_at" in manifest["timestamps"]
    assert "updated_at" in manifest["timestamps"]


def test_uninstall_dry_run_before_init(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["uninstall", "--dry-run"])

    assert result.exit_code == 0
    assert "Manifest: not found" in result.output
    assert "Files to remove: none" in result.output
    assert "No changes made." in result.output


def test_uninstall_purge_removes_only_manifest_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "app.py"
    source.write_text("def keep_me():\n    return True\n", encoding="utf-8")

    assert runner.invoke(app, ["init"]).exit_code == 0
    assert runner.invoke(app, ["mcp", "install", "--codex"]).exit_code == 0
    result = runner.invoke(app, ["uninstall", "--purge"])

    assert result.exit_code == 0
    assert source.exists()
    assert not (tmp_path / ".neurograph" / "brain.duckdb").exists()
    assert not (tmp_path / ".codex" / "config.toml").exists()
    assert not (tmp_path / ".neurograph").exists()


def test_uninstall_dry_run_reports_codex_config_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert runner.invoke(app, ["init"]).exit_code == 0
    assert runner.invoke(app, ["mcp", "install", "--codex"]).exit_code == 0
    result = runner.invoke(app, ["uninstall", "--dry-run"])

    assert result.exit_code == 0
    assert "Files to remove:" in result.output
    assert "Directories to remove:" in result.output
    assert "- .codex" in result.output
    assert "- .codex/config.toml" in result.output
    assert "Config blocks to edit:" in result.output
    assert "codex:.codex/config.toml#neurograph" in result.output


def test_uninstall_purge_refuses_without_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app_dir = tmp_path / ".neurograph"
    app_dir.mkdir()
    db = app_dir / "brain.duckdb"
    db.write_text("keep", encoding="utf-8")

    result = runner.invoke(app, ["uninstall", "--purge"])

    assert result.exit_code == 1
    assert "install-manifest.json is missing" in result.output
    assert db.exists()


def test_repeated_purge_refuses_safely_after_clean_purge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert runner.invoke(app, ["init"]).exit_code == 0
    assert runner.invoke(app, ["uninstall", "--purge"]).exit_code == 0
    second = runner.invoke(app, ["uninstall", "--purge"])

    assert second.exit_code == 1
    assert "install-manifest.json is missing" in second.output


def test_uninstall_keep_db_removes_cache_and_mcp_but_keeps_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert runner.invoke(app, ["init"]).exit_code == 0
    cache_file = tmp_path / ".neurograph" / "cache" / "nested" / "data.tmp"
    cache_file.parent.mkdir()
    cache_file.write_text("cache", encoding="utf-8")
    assert runner.invoke(app, ["mcp", "install", "--codex"]).exit_code == 0

    result = runner.invoke(app, ["uninstall", "--keep-db"])

    assert result.exit_code == 0
    assert (tmp_path / ".neurograph" / "brain.duckdb").exists()
    assert (tmp_path / ".neurograph" / "install-manifest.json").exists()
    assert not (tmp_path / ".neurograph" / "cache").exists()
    assert not (tmp_path / ".codex" / "config.toml").exists()


def test_preexisting_neurographignore_is_not_removed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ignore = tmp_path / ".neurographignore"
    ignore.write_text("custom\n", encoding="utf-8")

    assert runner.invoke(app, ["init"]).exit_code == 0
    result = runner.invoke(app, ["uninstall", "--purge"])

    assert result.exit_code == 0
    assert ignore.exists()
    assert ignore.read_text(encoding="utf-8") == "custom\n"


def test_unrecorded_file_in_neurograph_is_not_deleted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert runner.invoke(app, ["init"]).exit_code == 0
    user_file = tmp_path / ".neurograph" / "user-note.txt"
    user_file.write_text("do not remove\n", encoding="utf-8")
    dry_run = runner.invoke(app, ["uninstall", "--dry-run"])

    assert dry_run.exit_code == 0
    assert "Skipped for safety:" in dry_run.output
    assert ".neurograph (contains unrecorded .neurograph/user-note.txt)" in dry_run.output
    result = runner.invoke(app, ["uninstall", "--purge"])

    assert result.exit_code == 0
    assert user_file.exists()


def test_status_reports_required_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init"]).exit_code == 0

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Storage path:" in result.output
    assert "Indexed files:" in result.output
    assert "Changed files:" in result.output
    assert "MCP install status:" in result.output
    assert "External API: disabled" in result.output
