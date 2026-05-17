"""Typer CLI for NeuroGraph."""

from __future__ import annotations

from pathlib import Path

import typer

from neurograph import storage
from neurograph.graph.ask import ask_project
from neurograph.indexer.planner import run_index
from neurograph.lifecycle import (
    ManifestMissingError,
    build_uninstall_plan,
    init_project,
    is_initialized,
    load_manifest,
    purge,
    render_uninstall_plan,
    save_manifest,
    status as status_payload,
)
from neurograph.mcp.installer import (
    install_agent_skill,
    install_mcp_client,
    plan_mcp_install,
    plan_skill_install,
    render_mcp_install_preview,
    render_skill_install_preview,
    validate_agent_skill_install,
)


app = typer.Typer(help="NeuroGraph local context engine.", no_args_is_help=True)
mcp_app = typer.Typer(help="MCP integration commands.", no_args_is_help=True)
app.add_typer(mcp_app, name="mcp")


@app.command("init")
def init_command() -> None:
    """Initialize local NeuroGraph storage."""

    root = Path.cwd()
    result = init_project(root)
    typer.echo(f"Initialized NeuroGraph at {result.storage_path}")
    if result.created:
        typer.echo("Created:")
        for item in result.created:
            typer.echo(f"- {item}")
    else:
        typer.echo("Already initialized.")


@app.command("index")
def index_command(scip: Path | None = typer.Option(None, "--scip", help="Optional SCIP index path.")) -> None:
    """Index local code, Markdown, PDF, OpenAPI, SQL, and config text."""

    root = Path.cwd()
    if not is_initialized(root):
        typer.echo("NeuroGraph is not initialized. Run `ng init` first.", err=True)
        raise typer.Exit(1)
    storage.initialize(root)
    result = run_index(root, scip_path=scip)
    typer.echo(f"Total discovered: {result.total_discovered}")
    typer.echo(f"Skipped ignored: {result.skipped_ignored}")
    typer.echo(f"Unchanged: {result.skipped}")
    typer.echo(f"Changed/new indexed: {result.indexed}")
    typer.echo(f"Removed missing: {result.removed}")
    typer.echo(f"Markdown count: {result.kind_counts.get('markdown', 0)}")
    typer.echo(f"PDF count: {result.kind_counts.get('pdf', 0)}")
    typer.echo(f"Code count: {result.kind_counts.get('code', 0)}")
    typer.echo(f"OpenAPI count: {result.kind_counts.get('openapi', 0)}")
    typer.echo(f"SQL count: {result.kind_counts.get('sql', 0)}")
    typer.echo(f"Config count: {result.kind_counts.get('config', 0)}")
    typer.echo("PDF OCR/deep parse: off")
    if result.scip_status == "imported":
        typer.echo(
            "SCIP overlay: "
            f"imported symbols={result.scip_imported_symbols} "
            f"edges={result.scip_imported_edges} "
            f"path={result.scip_path}"
        )
    elif result.scip_path:
        detail = f" ({result.scip_message})" if result.scip_message else ""
        typer.echo(f"SCIP overlay: skipped status={result.scip_status} path={result.scip_path}{detail}")
    else:
        typer.echo("SCIP overlay: missing")
    typer.echo(f"Code-doc links: {result.code_doc_links}")
    typer.echo(f"Semantic candidates: {result.semantic_candidates}")
    typer.echo(f"Cost: {result.cost}")


@app.command("ask")
def ask_command(
    question: str = typer.Argument("", help="Coding task or project question."),
    token_budget: int = typer.Option(1800, "--token-budget", help="Approximate Context Pack token budget."),
    mode: str = typer.Option("auto", "--mode", help="Retrieval mode: auto, impact, why, conflict, implementation."),
) -> None:
    """Print a deterministic evidence-backed answer summary."""

    root = Path.cwd()
    if not is_initialized(root):
        typer.echo("NeuroGraph is not initialized. Run `ng init` and `ng index` first.", err=True)
        raise typer.Exit(1)
    _, summary = ask_project(root, question, token_budget=token_budget, mode=mode)
    typer.echo(summary, nl=False)


@app.command("status")
def status_command() -> None:
    """Print local NeuroGraph status."""

    data = status_payload(Path.cwd())
    counts = data["indexed_counts"]
    changes = data["changed_files"]
    typer.echo(f"Storage path: {data['storage_path']}")
    typer.echo(f"Initialized: {'yes' if data['initialized'] else 'no'}")
    typer.echo(
        "Indexed files: "
        f"total={counts.get('total', 0)} "
        f"code={counts.get('code', 0)} "
        f"markdown={counts.get('markdown', 0)} "
        f"pdf={counts.get('pdf', 0)} "
        f"sb={counts.get('sb', 0)}"
    )
    changed_total = sum(len(paths) for paths in changes.values())
    typer.echo(f"Changed files: {changed_total}")
    for key in ("new", "modified", "deleted"):
        if changes.get(key):
            typer.echo(f"  {key}: {', '.join(changes[key])}")
    mcp_clients = data.get("mcp_clients") or {}
    mcp_parts = []
    for key, label in (("codex", "codex"), ("claude_code", "claude-code"), ("gemini", "gemini")):
        item = mcp_clients.get(key) or {}
        if item.get("installed"):
            mcp_parts.append(f"{label}=installed ({item.get('config_path')})")
        else:
            mcp_parts.append(f"{label}=not installed")
    if not mcp_parts:
        if data["mcp_codex_installed"]:
            mcp_parts.append(f"codex=installed ({data['mcp_codex_config_path']})")
        else:
            mcp_parts.append("codex=not installed")
    typer.echo(f"MCP install status: {'; '.join(mcp_parts)}")
    skill_parts = []
    for key, label in (("codex", "codex"), ("claude_code", "claude-code"), ("gemini", "gemini")):
        item = mcp_clients.get(key) or {}
        if item.get("skill_installed"):
            skill_parts.append(f"{label}=installed ({item.get('skill_path')})")
        else:
            skill_parts.append(f"{label}=not installed")
    typer.echo(f"Agent skill status: {'; '.join(skill_parts)}")
    typer.echo(f"External API: {'enabled' if data['external_api_enabled'] else 'disabled'}")


@mcp_app.command("install")
def mcp_install_command(
    codex: bool = typer.Option(False, "--codex", help="Install Codex MCP config."),
    claude_code: bool = typer.Option(False, "--claude-code", help="Install Claude Code MCP config."),
    gemini: bool = typer.Option(False, "--gemini", help="Install Gemini CLI MCP config."),
    scope: str = typer.Option("project", "--scope", help="MCP config scope: project or user."),
    with_skill: bool = typer.Option(True, "--with-skill/--no-skill", help="Install agent skill instructions alongside MCP config."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show changes without writing config."),
) -> None:
    """Install read-only MCP integration."""

    clients = []
    if codex:
        clients.append("codex")
    if claude_code:
        clients.append("claude_code")
    if gemini:
        clients.append("gemini")
    if not clients:
        typer.echo("Specify --codex, --claude-code, or --gemini.", err=True)
        raise typer.Exit(1)

    root = Path.cwd()
    if with_skill:
        try:
            for client in clients:
                validate_agent_skill_install(root, client, scope=scope)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc

    if dry_run:
        for index, client in enumerate(clients):
            if index:
                typer.echo()
            try:
                preview = plan_mcp_install(root, client, scope=scope)
            except ValueError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1) from exc
            typer.echo(render_mcp_install_preview(preview), nl=False)
            if with_skill:
                typer.echo()
                typer.echo(render_skill_install_preview(plan_skill_install(root, client, scope=scope)), nl=False)
        return

    if not is_initialized(root):
        init_project(root)
    manifest = load_manifest(root)
    if manifest is None:
        typer.echo("Failed to load install manifest.", err=True)
        raise typer.Exit(1)
    for client in clients:
        try:
            config_path, created = install_mcp_client(root, manifest, client, scope=scope)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        action = "Created" if created else "Updated"
        label = {"codex": "Codex", "claude_code": "Claude Code", "gemini": "Gemini CLI"}[client]
        typer.echo(f"{action} {label} MCP config: {config_path}")
        if with_skill:
            try:
                skill_path, skill_created = install_agent_skill(root, manifest, client, scope=scope)
            except ValueError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1) from exc
            skill_action = "Created" if skill_created else "Updated"
            typer.echo(f"{skill_action} {label} agent skill: {skill_path}")
    save_manifest(root, manifest)
    suffix = " and agent skill." if with_skill else "."
    typer.echo(f"Installed read-only NeuroGraph MCP server{suffix}")


@mcp_app.command("serve")
def mcp_serve_command() -> None:
    """Run the read-only MCP server over stdio."""

    from neurograph.mcp.server import main as mcp_main

    mcp_main()


@app.command("uninstall")
def uninstall_command(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print uninstall plan without changing files."),
    purge_flag: bool = typer.Option(False, "--purge", help="Remove only manifest-recorded NeuroGraph files/config blocks."),
    keep_db: bool = typer.Option(False, "--keep-db", help="Remove MCP config and caches but keep .neurograph/brain.duckdb."),
) -> None:
    """Uninstall NeuroGraph project files safely."""

    root = Path.cwd()
    selected = sum(1 for value in (dry_run, purge_flag, keep_db) if value)
    if selected > 1:
        typer.echo("Choose only one of --dry-run, --purge, or --keep-db.", err=True)
        raise typer.Exit(1)
    if selected == 0:
        typer.echo("Specify --dry-run, --purge, or --keep-db.", err=True)
        raise typer.Exit(1)

    if dry_run:
        typer.echo(render_uninstall_plan(build_uninstall_plan(root)), nl=False)
        return

    try:
        removed = purge(root, keep_db=keep_db)
    except ManifestMissingError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if removed:
        heading = "Removed NeuroGraph-managed entries:" if keep_db else "Purged NeuroGraph-managed entries:"
        typer.echo(heading)
        for item in removed:
            typer.echo(f"- {item}")
    else:
        typer.echo("No NeuroGraph-managed entries found.")


def main() -> None:
    app()
