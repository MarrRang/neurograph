import json
import sys

from typer.testing import CliRunner

from neurograph.cli import app


runner = CliRunner()


def test_codex_install_creates_project_config_and_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")

    result = runner.invoke(app, ["mcp", "install", "--codex"])

    assert result.exit_code == 0
    config_path = tmp_path / ".codex" / "config.toml"
    assert config_path.is_file()
    text = config_path.read_text(encoding="utf-8")
    assert "# >>> neurograph mcp server" in text
    assert "[mcp_servers.neurograph]" in text
    assert f'command = "{sys.executable}"' in text
    assert 'args = ["-m", "neurograph.mcp.server"]' in text
    assert "NEUROGRAPH_PROJECT" in text

    manifest = json.loads((tmp_path / ".neurograph" / "install-manifest.json").read_text(encoding="utf-8"))
    codex = manifest["mcp"]["codex"]
    assert codex["installed"] is True
    assert codex["config_path"] == ".codex/config.toml"
    assert codex["config_format"] == "toml"
    assert codex["command"] == sys.executable
    assert codex["args"] == ["-m", "neurograph.mcp.server"]
    assert ".codex/config.toml" in codex["created_paths"]
    assert ".codex/config.toml" in codex["modified_paths"]
    assert ".codex/config.toml" in manifest["created_files"]
    assert ".codex" in manifest["created_directories"]
    assert "codex" in manifest["mcp_clients_touched"]
    assert manifest["inserted_config_blocks"][0]["path"] == ".codex/config.toml"
    assert manifest["inserted_config_blocks"][0]["block_key"] == "neurograph"


def test_codex_install_creates_project_agent_skill_and_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")

    result = runner.invoke(app, ["mcp", "install", "--codex"])
    status = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    skill_path = tmp_path / ".agents" / "skills" / "neurograph" / "SKILL.md"
    assert skill_path.is_file()
    skill_text = skill_path.read_text(encoding="utf-8")
    assert "name: neurograph" in skill_text
    assert "ng_context" in skill_text
    assert "untrusted evidence" in skill_text

    manifest = json.loads((tmp_path / ".neurograph" / "install-manifest.json").read_text(encoding="utf-8"))
    codex = manifest["mcp"]["codex"]
    assert codex["skill_installed"] is True
    assert codex["skill_path"] == ".agents/skills/neurograph/SKILL.md"
    assert ".agents/skills/neurograph/SKILL.md" in manifest["created_files"]
    assert ".agents/skills/neurograph" in manifest["created_directories"]
    assert status.exit_code == 0, status.output
    assert "Agent skill status: codex=installed (.agents/skills/neurograph/SKILL.md)" in status.output


def test_codex_install_dry_run_does_not_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")

    result = runner.invoke(app, ["mcp", "install", "--codex", "--dry-run"])

    assert result.exit_code == 0
    assert "NeuroGraph Codex MCP install dry run" in result.output
    assert "NeuroGraph Codex agent skill dry run" in result.output
    assert ".codex/config.toml" in result.output
    assert ".agents/skills/neurograph/SKILL.md" in result.output
    assert "No changes made." in result.output
    assert not (tmp_path / ".codex").exists()
    assert not (tmp_path / ".agents").exists()
    assert not (tmp_path / ".neurograph").exists()


def test_codex_user_scope_uses_mac_user_config_and_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(app, ["mcp", "install", "--codex", "--scope", "user"])
    status = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    config_path = home / ".codex" / "config.toml"
    assert config_path.is_file()
    assert not (tmp_path / ".codex").exists()
    text = config_path.read_text(encoding="utf-8")
    assert "[mcp_servers.neurograph]" in text
    assert f'command = "{sys.executable}"' in text
    assert "NEUROGRAPH_PROJECT" in text

    manifest = json.loads((tmp_path / ".neurograph" / "install-manifest.json").read_text(encoding="utf-8"))
    codex = manifest["mcp"]["codex"]
    assert codex["installed"] is True
    assert codex["scope"] == "user"
    assert codex["config_path"] == "~/.codex/config.toml"
    assert codex["created_config_file"] is True
    assert "~/.codex/config.toml" not in manifest["created_files"]
    assert any(block["client"] == "codex" and block["path"] == "~/.codex/config.toml" for block in manifest["inserted_config_blocks"])
    assert status.exit_code == 0, status.output
    assert "codex=installed (~/.codex/config.toml)" in status.output


def test_user_scope_install_dry_run_does_not_touch_home(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    result = runner.invoke(app, ["mcp", "install", "--gemini", "--scope", "user", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Scope: user" in result.output
    assert "~/.gemini/settings.json" in result.output
    assert "~/.agents/skills/neurograph/SKILL.md" in result.output
    assert not (home / ".gemini").exists()
    assert not (home / ".agents").exists()
    assert not (tmp_path / ".neurograph").exists()


def test_claude_code_install_creates_project_mcp_json_and_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")

    result = runner.invoke(app, ["mcp", "install", "--claude-code"])

    assert result.exit_code == 0, result.output
    config_path = tmp_path / ".mcp.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    server = data["mcpServers"]["neurograph"]
    assert server["command"] == sys.executable
    assert server["args"] == ["-m", "neurograph.mcp.server"]
    assert server["env"]["NEUROGRAPH_PROJECT"] == str(tmp_path.resolve())

    manifest = json.loads((tmp_path / ".neurograph" / "install-manifest.json").read_text(encoding="utf-8"))
    claude = manifest["mcp"]["claude_code"]
    assert claude["installed"] is True
    assert claude["config_path"] == ".mcp.json"
    assert claude["config_format"] == "json"
    assert ".mcp.json" in manifest["created_files"]
    assert "claude_code" in manifest["mcp_clients_touched"]
    assert any(block["client"] == "claude_code" and block["path"] == ".mcp.json" for block in manifest["inserted_config_blocks"])


def test_gemini_install_creates_project_settings_and_preserves_existing_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()
    config_path = gemini_dir / "settings.json"
    config_path.write_text(
        json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "other"}}}, indent=2),
        encoding="utf-8",
    )

    first = runner.invoke(app, ["mcp", "install", "--gemini"])
    second = runner.invoke(app, ["mcp", "install", "--gemini"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["theme"] == "dark"
    assert data["mcpServers"]["other"] == {"command": "other"}
    assert data["mcpServers"]["neurograph"]["args"] == ["mcp", "serve"]
    assert list(data["mcpServers"]).count("neurograph") == 1


def test_install_multiple_mcp_clients_and_status_reports_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["mcp", "install", "--codex", "--claude-code", "--gemini"])
    status = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".codex" / "config.toml").exists()
    assert (tmp_path / ".mcp.json").exists()
    assert (tmp_path / ".gemini" / "settings.json").exists()
    assert (tmp_path / ".agents" / "skills" / "neurograph" / "SKILL.md").exists()
    assert (tmp_path / ".claude" / "skills" / "neurograph" / "SKILL.md").exists()
    assert status.exit_code == 0, status.output
    assert "codex=installed (.codex/config.toml)" in status.output
    assert "claude-code=installed (.mcp.json)" in status.output
    assert "gemini=installed (.gemini/settings.json)" in status.output
    assert "codex=installed (.agents/skills/neurograph/SKILL.md)" in status.output
    assert "claude-code=installed (.claude/skills/neurograph/SKILL.md)" in status.output
    assert "gemini=installed (.agents/skills/neurograph/SKILL.md)" in status.output


def test_install_no_skill_leaves_agent_skill_uninstalled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["mcp", "install", "--codex", "--no-skill"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".codex" / "config.toml").exists()
    assert not (tmp_path / ".agents").exists()
    manifest = json.loads((tmp_path / ".neurograph" / "install-manifest.json").read_text(encoding="utf-8"))
    assert manifest["mcp"]["codex"]["skill_installed"] is False


def test_skill_install_refuses_to_overwrite_existing_non_neurograph_skill(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_path = tmp_path / ".agents" / "skills" / "neurograph" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("---\nname: neurograph\n---\ncustom\n", encoding="utf-8")

    result = runner.invoke(app, ["mcp", "install", "--codex"])

    assert result.exit_code == 1
    assert "Refusing to overwrite existing non-NeuroGraph skill" in result.output
    assert skill_path.read_text(encoding="utf-8") == "---\nname: neurograph\n---\ncustom\n"


def test_codex_install_preserves_existing_servers_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    config_path = codex_dir / "config.toml"
    config_path.write_text(
        '[mcp_servers.other]\ncommand = "other"\nargs = ["serve"]\n',
        encoding="utf-8",
    )

    first = runner.invoke(app, ["mcp", "install", "--codex"])
    second = runner.invoke(app, ["mcp", "install", "--codex"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    text = config_path.read_text(encoding="utf-8")
    assert '[mcp_servers.other]\ncommand = "other"\nargs = ["serve"]' in text
    assert text.count("[mcp_servers.neurograph]") == 1
    assert text.count("# >>> neurograph mcp server") == 1


def test_codex_install_backs_up_existing_config_before_edit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    config_path = codex_dir / "config.toml"
    config_path.write_text('[mcp_servers.other]\ncommand = "other"\n', encoding="utf-8")

    result = runner.invoke(app, ["mcp", "install", "--codex"])

    assert result.exit_code == 0
    manifest = json.loads((tmp_path / ".neurograph" / "install-manifest.json").read_text(encoding="utf-8"))
    backups = manifest["backups"]
    assert len(backups) == 1
    backup_path = tmp_path / backups[0]["backup_path"]
    assert backup_path.exists()
    assert backup_path.read_text(encoding="utf-8") == '[mcp_servers.other]\ncommand = "other"\n'
    assert backups[0]["source_path"] == ".codex/config.toml"
    assert backups[0]["reason"] == "mcp_install"


def test_uninstall_removes_only_neurograph_toml_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    config_path = codex_dir / "config.toml"
    config_path.write_text(
        '[mcp_servers.other]\ncommand = "other"\nargs = ["serve"]\n',
        encoding="utf-8",
    )

    assert runner.invoke(app, ["init"]).exit_code == 0
    assert runner.invoke(app, ["mcp", "install", "--codex"]).exit_code == 0
    result = runner.invoke(app, ["uninstall", "--purge"])

    assert result.exit_code == 0
    text = config_path.read_text(encoding="utf-8")
    assert "[mcp_servers.other]" in text
    assert "[mcp_servers.neurograph]" not in text
    assert "# >>> neurograph mcp server" not in text


def test_keep_db_backs_up_config_before_uninstall_edit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    config_path = codex_dir / "config.toml"
    config_path.write_text('[mcp_servers.other]\ncommand = "other"\n', encoding="utf-8")

    assert runner.invoke(app, ["init"]).exit_code == 0
    assert runner.invoke(app, ["mcp", "install", "--codex"]).exit_code == 0
    before = config_path.read_text(encoding="utf-8")
    result = runner.invoke(app, ["uninstall", "--keep-db"])

    assert result.exit_code == 0
    manifest = json.loads((tmp_path / ".neurograph" / "install-manifest.json").read_text(encoding="utf-8"))
    backups = [entry for entry in manifest["backups"] if entry["reason"] == "mcp_uninstall"]
    assert len(backups) == 1
    assert (tmp_path / backups[0]["backup_path"]).read_text(encoding="utf-8") == before


def test_uninstall_keeps_created_config_if_user_added_servers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert runner.invoke(app, ["mcp", "install", "--codex"]).exit_code == 0
    config_path = tmp_path / ".codex" / "config.toml"
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write('\n[mcp_servers.other]\ncommand = "other"\nargs = ["serve"]\n')

    dry_run = runner.invoke(app, ["uninstall", "--dry-run"])
    assert dry_run.exit_code == 0
    assert "codex:.codex/config.toml#neurograph" in dry_run.output
    assert "- .codex/config.toml" not in dry_run.output

    result = runner.invoke(app, ["uninstall", "--purge"])

    assert result.exit_code == 0
    text = config_path.read_text(encoding="utf-8")
    assert "[mcp_servers.other]" in text
    assert "[mcp_servers.neurograph]" not in text


def test_uninstall_removes_only_neurograph_json_servers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"other": {"command": "other"}}}, indent=2) + "\n",
        encoding="utf-8",
    )
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()
    (gemini_dir / "settings.json").write_text(
        json.dumps({"mcpServers": {"other": {"command": "other"}}, "theme": "dark"}, indent=2) + "\n",
        encoding="utf-8",
    )

    assert runner.invoke(app, ["init"]).exit_code == 0
    assert runner.invoke(app, ["mcp", "install", "--claude-code", "--gemini"]).exit_code == 0
    dry_run = runner.invoke(app, ["uninstall", "--dry-run"])
    result = runner.invoke(app, ["uninstall", "--purge"])

    assert dry_run.exit_code == 0, dry_run.output
    assert "claude_code:.mcp.json#neurograph" in dry_run.output
    assert "gemini:.gemini/settings.json#neurograph" in dry_run.output
    assert result.exit_code == 0, result.output
    claude_data = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    gemini_data = json.loads((gemini_dir / "settings.json").read_text(encoding="utf-8"))
    assert "neurograph" not in claude_data["mcpServers"]
    assert claude_data["mcpServers"]["other"] == {"command": "other"}
    assert "neurograph" not in gemini_data["mcpServers"]
    assert gemini_data["theme"] == "dark"


def test_user_scope_uninstall_removes_only_neurograph_entry_from_home_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config_path = home / ".claude.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"other": {"command": "other"}}, "theme": "dark"}, indent=2) + "\n",
        encoding="utf-8",
    )

    install = runner.invoke(app, ["mcp", "install", "--claude-code", "--scope", "user"])
    dry_run = runner.invoke(app, ["uninstall", "--dry-run"])
    purge = runner.invoke(app, ["uninstall", "--purge"])

    assert install.exit_code == 0, install.output
    assert dry_run.exit_code == 0, dry_run.output
    assert "claude_code:~/.claude.json#neurograph" in dry_run.output
    assert purge.exit_code == 0, purge.output
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["mcpServers"]["other"] == {"command": "other"}
    assert "neurograph" not in data["mcpServers"]
    assert data["theme"] == "dark"


def test_user_scope_uninstall_removes_config_file_created_by_neurograph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config_path = home / ".gemini" / "settings.json"

    install = runner.invoke(app, ["mcp", "install", "--gemini", "--scope", "user"])
    assert install.exit_code == 0, install.output
    assert config_path.exists()
    purge = runner.invoke(app, ["uninstall", "--purge"])

    assert purge.exit_code == 0, purge.output
    assert not config_path.exists()


def test_user_scope_uninstall_removes_user_agent_skill_created_by_neurograph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    skill_path = home / ".agents" / "skills" / "neurograph" / "SKILL.md"

    install = runner.invoke(app, ["mcp", "install", "--gemini", "--scope", "user"])
    dry_run = runner.invoke(app, ["uninstall", "--dry-run"])
    purge = runner.invoke(app, ["uninstall", "--purge"])

    assert install.exit_code == 0, install.output
    assert dry_run.exit_code == 0, dry_run.output
    assert "~/.agents/skills/neurograph/SKILL.md" in dry_run.output
    assert purge.exit_code == 0, purge.output
    assert not skill_path.exists()
    assert not (home / ".agents").exists()


def test_uninstall_removes_recorded_project_and_user_blocks_after_scope_switch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    project_install = runner.invoke(app, ["mcp", "install", "--codex"])
    user_install = runner.invoke(app, ["mcp", "install", "--codex", "--scope", "user"])
    dry_run = runner.invoke(app, ["uninstall", "--dry-run"])
    purge = runner.invoke(app, ["uninstall", "--purge"])

    assert project_install.exit_code == 0, project_install.output
    assert user_install.exit_code == 0, user_install.output
    assert "codex:.codex/config.toml#neurograph" in dry_run.output
    assert "codex:~/.codex/config.toml#neurograph" in dry_run.output
    assert purge.exit_code == 0, purge.output
    assert not (tmp_path / ".codex" / "config.toml").exists()
    assert not (home / ".codex" / "config.toml").exists()
    assert not (tmp_path / ".agents" / "skills" / "neurograph" / "SKILL.md").exists()
    assert not (home / ".agents" / "skills" / "neurograph" / "SKILL.md").exists()


def test_status_reports_codex_install_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert runner.invoke(app, ["init"]).exit_code == 0
    before = runner.invoke(app, ["status"])
    assert before.exit_code == 0
    assert "MCP install status: codex=not installed" in before.output

    assert runner.invoke(app, ["mcp", "install", "--codex"]).exit_code == 0
    after = runner.invoke(app, ["status"])
    assert after.exit_code == 0
    assert "MCP install status: codex=installed (.codex/config.toml)" in after.output
