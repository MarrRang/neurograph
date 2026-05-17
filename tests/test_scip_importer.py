import json

from typer.testing import CliRunner

from neurograph import storage
from neurograph.cli import app
from neurograph.indexer.planner import run_index
from neurograph.indexer.scip import EXACT_COMPILER, detect_scip_file, import_scip_overlay
from neurograph.lifecycle import init_project


runner = CliRunner()


def test_missing_scip_does_not_fail_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("def healthcheck():\n    return 'ok'\n", encoding="utf-8")

    assert runner.invoke(app, ["init"]).exit_code == 0
    result = runner.invoke(app, ["index"])

    assert result.exit_code == 0
    assert "SCIP overlay: missing" in result.output
    with storage.connect(tmp_path) as con:
        rows = con.execute("SELECT kind, label FROM nodes ORDER BY kind, label").fetchall()
    assert ("Function", "healthcheck") in rows


def test_imports_mocked_scip_json_as_exact_compiler_overlay(tmp_path):
    init_project(tmp_path)
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source = source_dir / "app.ts"
    source.write_text(
        "function createUser(req, res) {\n"
        "  return res.json({ ok: true });\n"
        "}\n"
        "app.post('/users', createUser);\n",
        encoding="utf-8",
    )
    scip = tmp_path / "index.scip"
    symbol = "scip-typescript local src/app.ts createUser()."
    scip.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "relative_path": "src/app.ts",
                        "occurrences": [
                            {
                                "symbol": symbol,
                                "range": [0, 9, 0, 19],
                                "symbol_roles": ["Definition"],
                                "syntax_kind": "Function",
                            },
                            {
                                "symbol": symbol,
                                "range": [3, 20, 3, 30],
                                "symbol_roles": ["Reference"],
                                "syntax_kind": "Function",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_index(tmp_path, scip)

    assert result.scip_status == "imported"
    assert result.scip_imported_symbols == 1
    assert result.scip_imported_edges == 2
    with storage.connect(tmp_path) as con:
        symbol_rows = con.execute(
            "SELECT kind, label, canonical_name, confidence FROM nodes WHERE kind = 'Symbol'"
        ).fetchall()
        edge_rows = con.execute(
            "SELECT relation, confidence, extractor FROM edges WHERE extractor = 'scip' ORDER BY relation"
        ).fetchall()
        linked_rows = con.execute(
            """
            SELECT n.kind, n.label, e.relation, s.kind, s.label
            FROM edges e
            JOIN nodes n ON n.id = e.src
            JOIN nodes s ON s.id = e.dst
            WHERE e.extractor = 'scip' AND e.relation = 'DEFINES'
            """
        ).fetchall()

    assert ("Symbol", "createUser", symbol, EXACT_COMPILER) in symbol_rows
    assert ("DEFINES", EXACT_COMPILER, "scip") in edge_rows
    assert ("REFERENCES", EXACT_COMPILER, "scip") in edge_rows
    assert ("Function", "createUser", "DEFINES", "Symbol", "createUser") in linked_rows


def test_detects_common_scip_location(tmp_path):
    scip_dir = tmp_path / ".scip"
    scip_dir.mkdir()
    scip = scip_dir / "index.scip"
    scip.write_text('{"documents": []}', encoding="utf-8")

    assert detect_scip_file(tmp_path) == scip.resolve()


def test_unsupported_binary_scip_is_skipped_safely(tmp_path):
    init_project(tmp_path)
    scip = tmp_path / "index.scip"
    scip.write_bytes(b"\x00\xff\x00not-json")

    result = import_scip_overlay(tmp_path, scip)

    assert result.status == "unsupported_binary"
    assert result.imported_symbols == 0
    assert result.imported_edges == 0
