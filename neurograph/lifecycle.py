"""Project lifecycle, manifest, status, and uninstall behavior."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from neurograph import config, storage
from neurograph.indexer.planner import changed_files
from neurograph.manifest import (
    normalize_manifest,
    now_iso,
    record_created_directory,
    record_created_file,
    record_modified_file,
    touch_manifest,
)
from neurograph.mcp.installer import (
    MCP_CLIENTS,
    agent_skill_is_installed,
    codex_is_installed,
    mcp_config_block_exists,
    mcp_client_is_installed,
    planned_agent_skill_paths,
    removable_created_paths_after_uninstall,
    remove_recorded_agent_skills,
    remove_mcp_config_block,
)
from neurograph.utils.ignore import default_ignore_text
from neurograph.utils.paths import (
    cache_dir,
    context_dir,
    db_path,
    ignore_path,
    manifest_path,
    neurograph_dir,
    resolve_recorded_path,
)


MANIFEST_REL_PATH = f"{config.APP_DIR_NAME}/{config.MANIFEST_NAME}"
DB_REL_PATH = f"{config.APP_DIR_NAME}/{config.DB_NAME}"
APP_OWNED_GENERATED_DIRS = {
    f"{config.APP_DIR_NAME}/{config.CACHE_DIR_NAME}",
    f"{config.APP_DIR_NAME}/{config.CONTEXT_DIR_NAME}",
    f"{config.APP_DIR_NAME}/backups",
}


@dataclass(frozen=True)
class InitResult:
    created: list[str]
    manifest_path: Path
    storage_path: Path


@dataclass(frozen=True)
class UninstallPlan:
    manifest_found: bool
    mode: str
    files: list[str]
    directories: list[str]
    mcp_blocks: list[str]
    skill_files: list[str]
    skipped: list[str]


class ManifestMissingError(RuntimeError):
    """Raised when a destructive lifecycle operation has no manifest."""


def default_manifest(root: Path) -> dict[str, Any]:
    return normalize_manifest(
        root,
        {
            "version": 1,
            "project_root": str(root.resolve()),
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "timestamps": {
                "created_at": now_iso(),
                "updated_at": now_iso(),
            },
            "created_files": [],
            "created_directories": [],
            "modified_files": [],
            "inserted_config_blocks": [],
            "backups": [],
            "mcp_clients_touched": [],
            "managed_paths": [],
            "mcp": {
                client: {
                    "installed": False,
                    "scope": "project",
                    "config_path": None,
                    "block_key": "neurograph",
                    "created_paths": [],
                    "created_config_file": False,
                    "created_config_dir": False,
                    "skill_installed": False,
                    "skill_path": None,
                    "created_skill_file": False,
                    "created_skill_dirs": [],
                    "skill_paths": [],
                }
                for client in MCP_CLIENTS
            },
        },
    )


def load_manifest(root: Path) -> dict[str, Any] | None:
    path = manifest_path(root)
    if not path.exists():
        return None
    return normalize_manifest(root, json.loads(path.read_text(encoding="utf-8")))


def save_manifest(root: Path, manifest: dict[str, Any]) -> None:
    manifest = normalize_manifest(root, manifest)
    if manifest_path(root).exists():
        record_modified_file(manifest, MANIFEST_REL_PATH)
    touch_manifest(manifest)
    manifest_path(root).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(root).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def init_project(root: Path) -> InitResult:
    root = root.resolve()
    created: list[str] = []
    manifest_existed = manifest_path(root).exists()
    db_existed = db_path(root).exists()
    manifest = load_manifest(root) or default_manifest(root)

    for path in (neurograph_dir(root), cache_dir(root), context_dir(root)):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            rel_path = path.relative_to(root).as_posix()
            record_created_directory(manifest, rel_path)
            created.append(rel_path)

    ignore = ignore_path(root)
    if not ignore.exists():
        ignore.write_text(default_ignore_text(), encoding="utf-8")
        rel_path = ignore.relative_to(root).as_posix()
        record_created_file(manifest, rel_path)
        created.append(rel_path)

    storage.initialize(root)
    if not db_existed:
        rel_path = db_path(root).relative_to(root).as_posix()
        record_created_file(manifest, rel_path)
        created.append(rel_path)
    if not manifest_existed:
        record_created_file(manifest, MANIFEST_REL_PATH)
    save_manifest(root, manifest)
    if not manifest_existed:
        created.append(manifest_path(root).relative_to(root).as_posix())

    return InitResult(created=created, manifest_path=manifest_path(root), storage_path=neurograph_dir(root))


def is_initialized(root: Path) -> bool:
    return manifest_path(root).exists() and db_path(root).exists()


def status(root: Path) -> dict[str, Any]:
    manifest = load_manifest(root)
    counts = storage.file_counts(root)
    changes = changed_files(root) if db_path(root).exists() else {"new": [], "modified": [], "deleted": []}
    codex = (manifest or {}).get("mcp", {}).get("codex", {})
    mcp_clients = {
        client: {
            "installed": mcp_client_is_installed(root, manifest, client),
            "config_path": (manifest or {}).get("mcp", {}).get(client, {}).get("config_path"),
            "skill_installed": agent_skill_is_installed(root, manifest, client),
            "skill_path": (manifest or {}).get("mcp", {}).get(client, {}).get("skill_path"),
        }
        for client in MCP_CLIENTS
    }
    codex_installed = codex_is_installed(root, manifest)
    return {
        "storage_path": str(neurograph_dir(root).resolve()),
        "initialized": is_initialized(root),
        "indexed_counts": counts,
        "changed_files": changes,
        "mcp_clients": mcp_clients,
        "mcp_codex_installed": codex_installed,
        "mcp_codex_config_path": codex.get("config_path"),
        "external_api_enabled": config.EXTERNAL_API_ENABLED,
    }


def build_uninstall_plan(root: Path, *, keep_db: bool = False) -> UninstallPlan:
    manifest = load_manifest(root)
    mode = "keep-db" if keep_db else "purge"
    if manifest is None:
        return UninstallPlan(
            manifest_found=False,
            mode=mode,
            files=[],
            directories=[],
            mcp_blocks=[],
            skill_files=[],
            skipped=[],
        )

    files, directories, skipped = _planned_removals(root, manifest, keep_db=keep_db)
    skill_files = planned_agent_skill_paths(root, manifest)
    blocks: list[str] = []
    for block in manifest.get("inserted_config_blocks", []):
        client = block.get("client")
        path = block.get("path")
        block_key = block.get("block_key", "neurograph")
        if client and path and mcp_config_block_exists(root, str(path), str(block_key)):
            blocks.append(f"{client}:{path}#{block_key}")
    return UninstallPlan(
        manifest_found=True,
        mode=mode,
        files=files,
        directories=directories,
        mcp_blocks=sorted(blocks),
        skill_files=skill_files,
        skipped=skipped,
    )


def render_uninstall_plan(plan: UninstallPlan) -> str:
    lines = ["NeuroGraph uninstall dry run"]
    lines.append(f"Mode: {plan.mode}")
    if not plan.manifest_found:
        lines.extend(
            [
                "Manifest: not found",
                "Files to remove: none",
                "Directories to remove: none",
                "Config blocks to edit: none",
                "Agent skills to remove: none",
                "No changes made.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.append("Manifest: found")
    lines.append("Files to remove:")
    if plan.files:
        lines.extend(f"- {path}" for path in plan.files)
    else:
        lines.append("- none")
    lines.append("Directories to remove:")
    if plan.directories:
        lines.extend(f"- {path}" for path in plan.directories)
    else:
        lines.append("- none")
    lines.append("Config blocks to edit:")
    if plan.mcp_blocks:
        lines.extend(f"- {block}" for block in plan.mcp_blocks)
    else:
        lines.append("- none")
    lines.append("Agent skills to remove:")
    if plan.skill_files:
        lines.extend(f"- {path}" for path in plan.skill_files)
    else:
        lines.append("- none")
    if plan.skipped:
        lines.append("Skipped for safety:")
        lines.extend(f"- {item}" for item in plan.skipped)
    lines.append("No changes made.")
    return "\n".join(lines) + "\n"


def purge(root: Path, *, keep_db: bool = False) -> list[str]:
    manifest = load_manifest(root)
    if manifest is None:
        raise ManifestMissingError(
            "Refusing destructive uninstall because .neurograph/install-manifest.json is missing."
        )

    removed: list[str] = []
    changed_any = False
    for block in list(manifest.get("inserted_config_blocks", [])):
        client = block.get("client")
        path = block.get("path")
        block_key = str(block.get("block_key") or "neurograph")
        if not client or not path:
            continue
        changed = remove_mcp_config_block(root, manifest, str(client), str(path), block_key)
        if changed:
            removed.append(f"mcp:{path}#{block_key}")
            changed_any = True

    for client in MCP_CLIENTS:
        manifest.get("mcp", {}).setdefault(client, {})["installed"] = False

    for skill_path in remove_recorded_agent_skills(root, manifest):
        removed.append(f"skill:{skill_path}")
        changed_any = True

    if changed_any:
        save_manifest(root, manifest)

    plan = build_uninstall_plan(root, keep_db=keep_db)
    for rel_path in sorted(plan.files, key=lambda item: (len(Path(item).parts), item), reverse=True):
        path = resolve_recorded_path(root, rel_path)
        if not path.exists() or not path.is_file():
            continue
        path.unlink()
        removed.append(rel_path)

    for rel_path in sorted(plan.directories, key=lambda item: (len(Path(item).parts), item), reverse=True):
        path = resolve_recorded_path(root, rel_path)
        if not path.exists() or not path.is_dir():
            continue
        try:
            path.rmdir()
            removed.append(rel_path)
        except OSError:
            continue

    if keep_db:
        save_manifest(root, manifest)

    return removed


def _planned_removals(root: Path, manifest: dict[str, Any], *, keep_db: bool) -> tuple[list[str], list[str], list[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    skipped: list[str] = []

    if keep_db:
        _add_owned_directory_tree(root, f"{config.APP_DIR_NAME}/{config.CACHE_DIR_NAME}", files, directories)
    else:
        for rel_path in manifest.get("created_files", []):
            if rel_path == DB_REL_PATH or rel_path.startswith(f"{config.APP_DIR_NAME}/"):
                files.add(rel_path)
            elif rel_path == config.IGNORE_FILE_NAME:
                files.add(rel_path)
        for rel_path in APP_OWNED_GENERATED_DIRS:
            _add_owned_directory_tree(root, rel_path, files, directories)
        for rel_path in manifest.get("created_directories", []):
            if rel_path == config.APP_DIR_NAME or rel_path.startswith(f"{config.APP_DIR_NAME}/"):
                directories.add(rel_path)

    for rel_path in removable_created_paths_after_uninstall(root, manifest):
        path = resolve_recorded_path(root, rel_path)
        if path.is_dir():
            directories.add(rel_path)
        else:
            files.add(rel_path)

    if keep_db:
        files.discard(DB_REL_PATH)
        files.discard(MANIFEST_REL_PATH)
        directories.discard(config.APP_DIR_NAME)
        directories.discard(f"{config.APP_DIR_NAME}/{config.CONTEXT_DIR_NAME}")
    else:
        _add_unremovable_directory_skips(root, files, directories, skipped)

    existing_files = sorted(rel for rel in files if _safe_exists(root, rel, file=True))
    existing_dirs = sorted(
        (rel for rel in directories if _safe_exists(root, rel, directory=True)),
        key=lambda item: (len(Path(item).parts), item),
    )
    return existing_files, existing_dirs, skipped


def _add_owned_directory_tree(root: Path, rel_path: str, files: set[str], directories: set[str]) -> None:
    directory = resolve_recorded_path(root, rel_path)
    if not directory.exists() or not directory.is_dir():
        return
    for child in directory.rglob("*"):
        rel_child = child.relative_to(root).as_posix()
        if child.is_file():
            files.add(rel_child)
        elif child.is_dir():
            directories.add(rel_child)
    directories.add(rel_path)


def _add_unremovable_directory_skips(root: Path, files: set[str], directories: set[str], skipped: list[str]) -> None:
    planned = set(files) | set(directories)
    blocked: set[str] = set()
    for rel_path in sorted(directories, key=lambda item: (len(Path(item).parts), item), reverse=True):
        path = resolve_recorded_path(root, rel_path)
        if not path.exists() or not path.is_dir():
            continue
        for child in path.iterdir():
            rel_child = child.relative_to(root).as_posix()
            if rel_child not in planned:
                skipped.append(f"{rel_path} (contains unrecorded {rel_child})")
                blocked.add(rel_path)
                break
    directories.difference_update(blocked)


def _safe_exists(root: Path, rel_path: str, *, file: bool = False, directory: bool = False) -> bool:
    try:
        path = resolve_recorded_path(root, rel_path)
    except ValueError:
        return False
    if file:
        return path.is_file()
    if directory:
        return path.is_dir()
    return path.exists()


def _config_block_exists(root: Path, rel_path: str, block_key: str) -> bool:
    try:
        path = resolve_recorded_path(root, rel_path)
    except ValueError:
        return False
    if not path.exists() or not path.is_file():
        return False
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            return False
        servers = data.get("mcpServers")
        return isinstance(servers, dict) and block_key in servers
    return f"[mcp_servers.{block_key}]" in path.read_text(encoding="utf-8")
