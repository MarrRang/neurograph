"""Install manifest helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIST_KEYS = (
    "created_files",
    "created_directories",
    "modified_files",
    "inserted_config_blocks",
    "backups",
    "mcp_clients_touched",
    "managed_paths",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Backfill v0.1 lifecycle fields while preserving older manifest keys."""

    now = now_iso()
    manifest.setdefault("version", 1)
    manifest.setdefault("project_root", str(root.resolve()))
    created_at = str(manifest.get("created_at") or manifest.get("timestamps", {}).get("created_at") or now)
    manifest["created_at"] = created_at
    manifest.setdefault("updated_at", now)
    timestamps = manifest.setdefault("timestamps", {})
    timestamps.setdefault("created_at", created_at)
    timestamps.setdefault("updated_at", manifest["updated_at"])

    for key in LIST_KEYS:
        value = manifest.setdefault(key, [])
        if isinstance(value, list):
            if key in {"inserted_config_blocks", "backups"}:
                continue
            manifest[key] = sorted(dict.fromkeys(str(item) for item in value))
        else:
            manifest[key] = []

    _migrate_managed_paths(manifest)
    _normalize_mcp(manifest)
    return manifest


def touch_manifest(manifest: dict[str, Any]) -> None:
    updated_at = now_iso()
    manifest["updated_at"] = updated_at
    timestamps = manifest.setdefault("timestamps", {})
    timestamps.setdefault("created_at", manifest.get("created_at", updated_at))
    timestamps["updated_at"] = updated_at


def record_created_file(manifest: dict[str, Any], rel_path: str) -> None:
    _add_path(manifest, "created_files", rel_path)
    _add_path(manifest, "managed_paths", rel_path)


def record_created_directory(manifest: dict[str, Any], rel_path: str) -> None:
    _add_path(manifest, "created_directories", rel_path)
    _add_path(manifest, "managed_paths", rel_path)


def record_modified_file(manifest: dict[str, Any], rel_path: str) -> None:
    _add_path(manifest, "modified_files", rel_path)


def record_mcp_client(manifest: dict[str, Any], client: str) -> None:
    _add_path(manifest, "mcp_clients_touched", client)


def record_inserted_config_block(
    manifest: dict[str, Any],
    *,
    client: str,
    path: str,
    block_key: str,
    start_marker: str,
    end_marker: str,
    config_format: str,
) -> None:
    blocks = manifest.setdefault("inserted_config_blocks", [])
    now = now_iso()
    for block in blocks:
        if block.get("client") == client and block.get("path") == path and block.get("block_key") == block_key:
            block.update(
                {
                    "start_marker": start_marker,
                    "end_marker": end_marker,
                    "config_format": config_format,
                    "updated_at": now,
                }
            )
            return
    blocks.append(
        {
            "client": client,
            "path": path,
            "block_key": block_key,
            "start_marker": start_marker,
            "end_marker": end_marker,
            "config_format": config_format,
            "created_at": now,
            "updated_at": now,
        }
    )


def record_backup(
    manifest: dict[str, Any],
    *,
    client: str,
    source_path: str,
    backup_path: str,
    reason: str,
) -> None:
    manifest.setdefault("backups", []).append(
        {
            "client": client,
            "source_path": source_path,
            "backup_path": backup_path,
            "reason": reason,
            "created_at": now_iso(),
        }
    )
    record_created_file(manifest, backup_path)


def _add_path(manifest: dict[str, Any], key: str, rel_path: str) -> None:
    values = manifest.setdefault(key, [])
    normalized = str(Path(rel_path).as_posix()).rstrip("/")
    if normalized not in values:
        values.append(normalized)
        values.sort()


def _migrate_managed_paths(manifest: dict[str, Any]) -> None:
    if manifest.get("created_files") or manifest.get("created_directories"):
        return
    for rel_path in manifest.get("managed_paths", []):
        path = Path(str(rel_path))
        if path.suffix or path.name in {"brain.duckdb", "install-manifest.json", ".neurographignore"}:
            _add_path(manifest, "created_files", str(rel_path))
        else:
            _add_path(manifest, "created_directories", str(rel_path))


def _normalize_mcp(manifest: dict[str, Any]) -> None:
    mcp = manifest.setdefault("mcp", {})
    defaults = {
        "codex": {"config_format": "toml"},
        "claude_code": {"config_format": "json"},
        "gemini": {"config_format": "json"},
    }
    for client, values in defaults.items():
        state = mcp.setdefault(
            client,
            {
                "installed": False,
                "config_path": None,
                "block_key": "neurograph",
                "created_paths": [],
            },
        )
        state.setdefault("installed", False)
        state.setdefault("config_path", None)
        state.setdefault("block_key", "neurograph")
        state.setdefault("created_paths", [])
        state.setdefault("modified_paths", [])
        state.setdefault("config_format", values["config_format"])
        state.setdefault("scope", "user" if str(state.get("config_path") or "").startswith("~/") else "project")
        state.setdefault("created_config_file", False)
        state.setdefault("created_config_dir", False)
        state.setdefault("skill_installed", False)
        state.setdefault("skill_scope", "user" if str(state.get("skill_path") or "").startswith("~/") else "project")
        state.setdefault("skill_path", None)
        state.setdefault("created_skill_file", False)
        state.setdefault("created_skill_dirs", [])
        state.setdefault("skill_paths", [])

        if state.get("installed") and state.get("config_path"):
            config_format = state.get("config_format") or values["config_format"]
            record_mcp_client(manifest, client)
            record_inserted_config_block(
                manifest,
                client=client,
                path=state["config_path"],
                block_key=state.get("block_key") or "neurograph",
                start_marker=(state.get("block_markers") or {}).get("start", "# >>> neurograph mcp server" if config_format == "toml" else ""),
                end_marker=(state.get("block_markers") or {}).get("end", "# <<< neurograph mcp server" if config_format == "toml" else ""),
                config_format=config_format,
            )
