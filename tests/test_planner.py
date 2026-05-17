from neurograph import storage
from neurograph.indexer.planner import (
    build_index_plan,
    classify_file,
    discover_files,
    file_hash,
    should_index,
)


def test_classify_file_supported_kinds_and_media_exclusion(tmp_path):
    assert classify_file(tmp_path / "README.md") == "markdown"
    assert classify_file(tmp_path / "guide.markdown") == "markdown"
    assert classify_file(tmp_path / "paper.pdf") == "pdf"
    assert classify_file(tmp_path / "src" / "main.py") == "code"
    assert classify_file(tmp_path / "openapi.yaml") == "openapi"
    assert classify_file(tmp_path / "schema.sql") == "sql"
    assert classify_file(tmp_path / "schema.graphql") == "config"
    assert classify_file(tmp_path / "settings.yaml") == "config"
    assert classify_file(tmp_path / "photo.png") is None
    assert classify_file(tmp_path / "song.mp3") is None
    assert classify_file(tmp_path / "movie.mp4") is None


def test_discover_files_is_deterministic_and_respects_ignore_files(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored-by-git.md\nprivate/\n", encoding="utf-8")
    (tmp_path / ".neurographignore").write_text("ignored-by-neurograph.md\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b():\n    pass\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / "ignored-by-git.md").write_text("# ignored\n", encoding="utf-8")
    (tmp_path / "ignored-by-neurograph.md").write_text("# ignored\n", encoding="utf-8")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "keep.md").write_text("# ignored\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.md").write_text("# ignored\n", encoding="utf-8")

    first = [path.relative_to(tmp_path).as_posix() for path in discover_files(tmp_path)]
    second = [path.relative_to(tmp_path).as_posix() for path in discover_files(tmp_path)]

    assert first == second == ["a.md", "b.py"]
    assert should_index(tmp_path / "a.md", root=tmp_path)
    assert not should_index(tmp_path / "ignored-by-neurograph.md", root=tmp_path)
    assert not should_index(tmp_path / "node_modules" / "pkg.md", root=tmp_path)


def test_build_index_plan_hashes_and_skips_unchanged(tmp_path):
    storage.init_db(tmp_path)
    readme = tmp_path / "README.md"
    api = tmp_path / "openapi.yaml"
    lock = tmp_path / "package.lock"
    readme.write_text("# NeuroGraph\n", encoding="utf-8")
    api.write_text("openapi: 3.1.0\n", encoding="utf-8")
    lock.write_text("ignored lockfile\n", encoding="utf-8")

    storage.upsert_artifact(
        tmp_path,
        id="artifact-readme",
        path="README.md",
        kind="markdown",
        content_hash=file_hash(readme),
    )

    index_plan = build_index_plan(tmp_path)

    assert index_plan.total_discovered == 3
    assert index_plan.skipped_ignored == 2
    assert index_plan.skipped_unsupported == 0
    assert [target.rel_path for target in index_plan.unchanged] == ["README.md"]
    assert [target.rel_path for target in index_plan.changed] == ["openapi.yaml"]
    assert index_plan.kind_counts == {"markdown": 1, "openapi": 1}
