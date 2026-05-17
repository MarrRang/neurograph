from neurograph import storage
from neurograph.graph.builder import build_file_index
from neurograph.indexer.code import EXACT_STATIC, index_code_graph
from neurograph.utils.hashing import sha256_file


def test_typescript_graph_extracts_imports_classes_functions_methods_and_routes(tmp_path):
    source = tmp_path / "app.ts"
    source.write_text(
        "import express from 'express';\n"
        "const app = express();\n"
        "function createUser(req, res) {\n"
        "  return res.json({ ok: true });\n"
        "}\n"
        "class UserService {\n"
        "  saveUser() {\n"
        "    return true;\n"
        "  }\n"
        "}\n"
        "app.post('/users', createUser);\n",
        encoding="utf-8",
    )

    graph = index_code_graph(source, "src/app.ts", "code")
    nodes = _nodes_by_kind(graph.nodes)
    relations = {(edge.relation, _label(graph, edge.source), _label(graph, edge.target)) for edge in graph.edges}

    assert any(node.label == "src/app.ts" for node in nodes["File"])
    assert any(node.label == "express" for node in nodes["Module"])
    assert any(node.label == "UserService" for node in nodes["Class"])
    assert any(node.label == "createUser" for node in nodes["Function"])
    assert any(node.label == "saveUser" for node in nodes["Method"])
    assert any(node.label == "POST /users" for node in nodes["Endpoint"])
    assert ("IMPORTS", "src/app.ts", "express") in relations
    assert ("ROUTE_TO_HANDLER", "POST /users", "createUser") in relations
    assert _all_nodes_have_line_evidence(graph.nodes)


def test_python_graph_uses_ast_and_survives_syntax_errors(tmp_path):
    source = tmp_path / "api.py"
    source.write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.get('/items')\n"
        "def list_items():\n"
        "    return []\n"
        "class Repo:\n"
        "    def save(self):\n"
        "        return True\n",
        encoding="utf-8",
    )

    graph = index_code_graph(source, "api.py", "code")
    nodes = _nodes_by_kind(graph.nodes)
    relations = {(edge.relation, _label(graph, edge.source), _label(graph, edge.target)) for edge in graph.edges}

    assert any(node.label == "fastapi" for node in nodes["Module"])
    assert any(node.label == "Repo" for node in nodes["Class"])
    assert any(node.label == "list_items" for node in nodes["Function"])
    assert any(node.label == "save" for node in nodes["Method"])
    assert any(node.label == "GET /items" for node in nodes["Endpoint"])
    assert ("ROUTE_TO_HANDLER", "GET /items", "list_items") in relations

    broken = tmp_path / "broken.py"
    broken.write_text("def bad(:\n    pass\nclass StillIndexed:\n    pass\n", encoding="utf-8")
    broken_graph = index_code_graph(broken, "broken.py", "code")

    assert any(node.kind == "Class" and node.label == "StillIndexed" for node in broken_graph.nodes)
    assert _all_nodes_have_line_evidence(broken_graph.nodes)


def test_java_graph_extracts_imports_class_method_and_spring_route(tmp_path):
    source = tmp_path / "UserController.java"
    source.write_text(
        "package com.acme;\n"
        "import java.util.List;\n"
        "class UserController {\n"
        "  @GetMapping(\"/users\")\n"
        "  public String listUsers() {\n"
        "    return \"ok\";\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    graph = index_code_graph(source, "src/UserController.java", "code")
    nodes = _nodes_by_kind(graph.nodes)
    relations = {(edge.relation, _label(graph, edge.source), _label(graph, edge.target)) for edge in graph.edges}

    assert any(node.label == "com.acme" for node in nodes["Module"])
    assert any(node.label == "java.util.List" for node in nodes["Module"])
    assert any(node.label == "UserController" for node in nodes["Class"])
    assert any(node.label == "listUsers" for node in nodes["Method"])
    assert any(node.label == "GET /users" for node in nodes["Endpoint"])
    assert ("ROUTE_TO_HANDLER", "GET /users", "listUsers") in relations


def test_sql_graph_extracts_tables_columns_and_query_touch_edges(tmp_path):
    source = tmp_path / "schema.sql"
    source.write_text(
        "CREATE TABLE users (\n"
        "  id integer,\n"
        "  email text,\n"
        "  status text\n"
        ");\n"
        "SELECT id, email FROM users WHERE status = 'ACTIVE';\n",
        encoding="utf-8",
    )

    graph = index_code_graph(source, "db/schema.sql", "sql")
    nodes = _nodes_by_kind(graph.nodes)
    relations = {(edge.relation, _label(graph, edge.source), _label(graph, edge.target)) for edge in graph.edges}

    assert any(node.label == "users" for node in nodes["Table"])
    assert {"id", "email", "status"}.issubset({node.label for node in nodes["Column"]})
    assert ("DEFINES", "db/schema.sql", "users") in relations
    assert ("QUERY_TOUCHES_TABLE", "db/schema.sql", "users") in relations


def test_openapi_and_config_graph_extract_endpoints_and_config_keys(tmp_path):
    yaml_source = tmp_path / "openapi.yaml"
    yaml_source.write_text(
        "openapi: 3.1.0\n"
        "paths:\n"
        "  /users:\n"
        "    get:\n"
        "      operationId: listUsers\n",
        encoding="utf-8",
    )
    json_source = tmp_path / "openapi.json"
    json_source.write_text(
        '{"openapi":"3.1.0","paths":{"/orders":{"post":{"operationId":"createOrder"}}}}',
        encoding="utf-8",
    )
    config_source = tmp_path / "app.yaml"
    config_source.write_text("server:\n  port: 3000\nfeature_flags:\n  checkout: true\n", encoding="utf-8")

    yaml_graph = index_code_graph(yaml_source, "openapi.yaml", "openapi")
    json_graph = index_code_graph(json_source, "openapi.json", "openapi")
    config_graph = index_code_graph(config_source, "app.yaml", "config")

    assert any(node.kind == "Endpoint" and node.label == "GET /users" for node in yaml_graph.nodes)
    assert any(node.kind == "Endpoint" and node.label == "POST /orders" for node in json_graph.nodes)
    assert any(node.kind == "ConfigKey" and node.canonical_name == "server" for node in config_graph.nodes)
    assert any(node.kind == "ConfigKey" and node.canonical_name == "checkout" for node in config_graph.nodes)


def test_code_graph_nodes_are_persisted_with_evidence(tmp_path):
    storage.init_db(tmp_path)
    source = tmp_path / "app.py"
    source.write_text("def healthcheck():\n    return 'ok'\n", encoding="utf-8")

    record, chunks, edges = build_file_index(tmp_path, source, "app.py", "code", sha256_file(source))
    storage.replace_file_index(tmp_path, record, chunks, edges)

    with storage.connect(tmp_path) as con:
        node_rows = con.execute("SELECT kind, label, start_line, end_line, confidence FROM nodes ORDER BY kind, label").fetchall()
        edge_rows = con.execute("SELECT relation FROM edges").fetchall()
        evidence_rows = con.execute("SELECT source_uri, start_line, end_line, confidence FROM evidence").fetchall()

    assert ("File", "app.py", 1, 2, EXACT_STATIC) in node_rows
    assert any(row[0] == "Function" and row[1] == "healthcheck" for row in node_rows)
    assert ("DEFINES",) in edge_rows
    assert all(row[0] == "app.py" and row[1] is not None and row[2] is not None and row[3] == EXACT_STATIC for row in evidence_rows)


def _nodes_by_kind(nodes):
    by_kind = {}
    for node in nodes:
        by_kind.setdefault(node.kind, []).append(node)
    return by_kind


def _label(graph, node_id):
    return next(node.label for node in graph.nodes if node.id == node_id)


def _all_nodes_have_line_evidence(nodes):
    return all(node.evidence.source_path and node.evidence.start_line >= 1 and node.evidence.end_line >= node.evidence.start_line and node.evidence.quote for node in nodes)
