from neurograph import storage
from neurograph.graph.schema import Chunk, FileRecord
from neurograph.lifecycle import init_project


def test_init_db_creates_requested_tables_idempotently(tmp_path):
    storage.init_db(tmp_path)
    storage.init_db(tmp_path)

    assert (tmp_path / ".neurograph" / "brain.duckdb").exists()
    with storage.connect(tmp_path) as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}

    assert {
        "artifacts",
        "nodes",
        "edges",
        "chunks",
        "evidence",
        "context_packs",
        "manifests",
    }.issubset(tables)
    assert storage.get_counts(tmp_path)["manifests"] == 1


def test_storage_upserts_updates_counts_and_changed_files(tmp_path):
    storage.init_db(tmp_path)

    storage.upsert_artifact(
        tmp_path,
        id="artifact-1",
        uri="docs/readme.md",
        path="docs/readme.md",
        kind="markdown",
        title="Readme",
        content_hash="hash-1",
        metadata={"source": "test"},
    )
    storage.upsert_chunk(
        tmp_path,
        id="chunk-1",
        artifact_id="artifact-1",
        kind="markdown",
        text="NeuroGraph builds context packs for Codex and Claude.",
        start_line=1,
        end_line=2,
        metadata={"evidence": "docs/readme.md:1-2"},
    )
    storage.upsert_evidence(
        tmp_path,
        id="evidence-1",
        artifact_id="artifact-1",
        source_uri="docs/readme.md",
        quote="NeuroGraph builds context packs",
        start_line=1,
        end_line=1,
        extractor="test",
        confidence="high",
    )
    storage.upsert_node(
        tmp_path,
        id="node-1",
        kind="section",
        label="Overview",
        artifact_id="artifact-1",
        path="docs/readme.md",
        start_line=1,
        end_line=2,
        confidence="high",
    )
    storage.upsert_edge(
        tmp_path,
        id="edge-1",
        src="artifact-1",
        dst="node-1",
        relation="contains",
        confidence="high",
        score=1.0,
        extractor="test",
        evidence_id="evidence-1",
    )

    counts = storage.get_counts(tmp_path)
    assert counts["artifacts"] == 1
    assert counts["nodes"] == 1
    assert counts["edges"] == 1
    assert counts["chunks"] == 1
    assert counts["evidence"] == 1
    assert counts["by_kind"] == {"markdown": 1}

    storage.upsert_artifact(
        tmp_path,
        id="artifact-1",
        uri="docs/readme.md",
        path="docs/readme.md",
        kind="markdown",
        title="Updated Readme",
        content_hash="hash-2",
    )
    assert storage.indexed_hashes(tmp_path) == {"docs/readme.md": "hash-2"}
    assert storage.get_counts(tmp_path)["artifacts"] == 1

    assert storage.get_changed_files(tmp_path, {"docs/readme.md": "hash-2"}) == {
        "new": [],
        "modified": [],
        "deleted": [],
    }
    assert storage.get_changed_files(tmp_path, {"docs/readme.md": "changed", "src/app.py": "new"}) == {
        "new": ["src/app.py"],
        "modified": ["docs/readme.md"],
        "deleted": [],
    }
    assert storage.get_changed_files(tmp_path, {}) == {
        "new": [],
        "modified": [],
        "deleted": ["docs/readme.md"],
    }

    assert storage.clear_artifact_index(tmp_path, "docs/readme.md") == 1
    counts = storage.get_counts(tmp_path)
    assert counts["artifacts"] == 0
    assert counts["nodes"] == 0
    assert counts["edges"] == 0
    assert counts["chunks"] == 0
    assert counts["evidence"] == 0


def test_search_text_falls_back_to_like_when_fts_unavailable(tmp_path, monkeypatch):
    storage.init_db(tmp_path)
    storage.upsert_artifact(
        tmp_path,
        id="artifact-1",
        path="README.md",
        kind="markdown",
        content_hash="hash-1",
    )
    storage.upsert_chunk(
        tmp_path,
        id="chunk-1",
        artifact_id="artifact-1",
        kind="markdown",
        text="Local context pack search should work without FTS.",
        start_line=1,
        end_line=1,
    )

    def fail_fts(con, query, limit):
        raise RuntimeError("fts unavailable")

    monkeypatch.setattr(storage, "_try_fts_search", fail_fts)

    hits = storage.search_text(tmp_path, "context pack", limit=5)

    assert len(hits) == 1
    assert hits[0]["id"] == "chunk-1"
    assert hits[0]["search_method"] == "like"


def test_storage_replaces_file_index(tmp_path):
    init_project(tmp_path)
    record = FileRecord(
        path="README.md",
        kind="markdown",
        sha256="abc",
        size_bytes=12,
        mtime=1.0,
        title="Readme",
    )
    chunk = Chunk(
        id="chunk-1",
        file_path="README.md",
        kind="markdown",
        heading="Intro",
        start_line=1,
        end_line=3,
        text="# Intro\nHello",
        evidence="README.md:1-3",
    )

    storage.replace_file_index(tmp_path, record, [chunk])

    assert storage.file_counts(tmp_path) == {"markdown": 1, "total": 1}
    assert storage.indexed_hashes(tmp_path) == {"README.md": "abc"}
    chunks = storage.all_chunks(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].evidence == "README.md:1-3"
