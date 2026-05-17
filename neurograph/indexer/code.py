"""Fast deterministic code graph extraction without build tools."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from neurograph.utils.hashing import sha256_text


EXACT_STATIC = "EXACT_STATIC"
AMBIGUOUS = "AMBIGUOUS"
EXTRACTOR = "code_fast_graph"

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
TS_IMPORT_RE = re.compile(r"^\s*import(?:\s+[^'\"]+\s+from)?\s*['\"]([^'\"]+)['\"]")
TS_REQUIRE_RE = re.compile(r"\brequire\(\s*['\"]([^'\"]+)['\"]\s*\)")
TS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)\b")
TS_FUNCTION_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(")
TS_ARROW_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>")
TS_METHOD_RE = re.compile(r"^\s*(?:public|private|protected|static|async|get|set|\s)*\s*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*[{:]")
EXPRESS_ROUTE_RE = re.compile(
    r"\b(?:app|router)\.(get|post|put|patch|delete|head|options)\s*\(\s*['\"]([^'\"]+)['\"]\s*(?:,\s*([A-Za-z_$][\w$]*))?",
    re.IGNORECASE,
)
PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)(?:\s+import\s+.+)|import\s+([A-Za-z_][\w.]*))")
PY_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][\w]*)\b")
PY_FUNC_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][\w]*)\s*\(")
JAVA_PACKAGE_RE = re.compile(r"^\s*package\s+([A-Za-z_][\w.]*);")
JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([A-Za-z_][\w.*]*);")
JAVA_CLASS_RE = re.compile(r"\b(?:class|interface|enum|record)\s+([A-Za-z_][\w]*)\b")
JAVA_METHOD_RE = re.compile(
    r"^\s*(?:public|private|protected|static|final|synchronized|abstract|native|\s)+"
    r"[A-Za-z_<>\[\], ?]+\s+([A-Za-z_][\w]*)\s*\([^;]*\)\s*(?:throws\s+[^{]+)?\{?"
)
JAVA_MAPPING_RE = re.compile(r"@(Get|Post|Put|Patch|Delete|Request)Mapping\s*(?:\(\s*(?:value\s*=\s*)?[\"']([^\"']+)[\"'])?", re.IGNORECASE)
SQL_TABLE_RE = re.compile(
    r"\b(?:CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?|ALTER\s+TABLE|INSERT\s+INTO|UPDATE|FROM|JOIN|DELETE\s+FROM)\s+([A-Za-z_][\w.]*|\"[^\"]+\"|`[^`]+`)",
    re.IGNORECASE,
)
SQL_CREATE_RE = re.compile(r"\bCREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+([A-Za-z_][\w.]*|\"[^\"]+\"|`[^`]+`)", re.IGNORECASE)
SQL_COLUMN_RE = re.compile(r"^\s*([A-Za-z_][\w]*|\"[^\"]+\"|`[^`]+`)\s+(?:[A-Za-z]+|[A-Z]+)\b")
YAML_KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_./{}:-]+)\s*:")
YAML_METHOD_RE = re.compile(r"^\s{2,}(get|post|put|patch|delete|head|options)\s*:\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class CodeEvidence:
    source_path: str
    start_line: int
    end_line: int
    quote: str
    confidence: str


@dataclass(frozen=True)
class CodeNode:
    id: str
    kind: str
    label: str
    canonical_name: str
    evidence: CodeEvidence
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodeEdge:
    source: str
    target: str
    relation: str
    evidence: CodeEvidence
    confidence: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodeGraph:
    source_path: str
    language: str
    nodes: tuple[CodeNode, ...]
    edges: tuple[CodeEdge, ...]


@dataclass(frozen=True)
class CodeBlock:
    symbol: str | None
    start_line: int
    end_line: int
    text: str


def index_code_graph(path: Path, source_path: str | None = None, kind: str | None = None) -> CodeGraph:
    source = source_path or path.as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    language = _language(path, kind)
    nodes: list[CodeNode] = []
    edges: list[CodeEdge] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    file_node = _node(source, "File", source, source, 1, max(1, len(lines)), _quote(lines, 1, min(len(lines), 3)), EXACT_STATIC, {"language": language})
    _add_node(nodes, seen_nodes, file_node)

    module_node: CodeNode | None = None
    if language in {"python", "typescript", "javascript", "java"}:
        module_name = _module_name(source, language, lines)
        module_node = _node(source, "Module", module_name, module_name, 1, 1, _quote(lines, 1, 1) or source, EXACT_STATIC, {"language": language})
        _add_node(nodes, seen_nodes, module_node)
        _add_edge(edges, seen_edges, file_node, module_node, "CONTAINS")

    if language == "python":
        _extract_python(source, lines, file_node, module_node or file_node, nodes, edges, seen_nodes, seen_edges)
    elif language in {"typescript", "javascript"}:
        _extract_ts_js(source, lines, file_node, module_node or file_node, nodes, edges, seen_nodes, seen_edges)
    elif language == "java":
        _extract_java(source, lines, file_node, module_node or file_node, nodes, edges, seen_nodes, seen_edges)
    elif language == "sql":
        _extract_sql(source, lines, file_node, nodes, edges, seen_nodes, seen_edges)
    elif language == "openapi":
        _extract_openapi(source, text, lines, file_node, nodes, edges, seen_nodes, seen_edges)
    elif language == "config":
        _extract_config(source, text, lines, file_node, nodes, edges, seen_nodes, seen_edges)

    return CodeGraph(source_path=source, language=language, nodes=tuple(nodes), edges=tuple(edges))


def parse_code(path: Path) -> list[CodeBlock]:
    """Compatibility wrapper returning symbol snippets."""

    graph = index_code_graph(path, path.as_posix(), "code")
    blocks: list[CodeBlock] = []
    for node in graph.nodes:
        if node.kind in {"Class", "Function", "Method"}:
            blocks.append(
                CodeBlock(
                    symbol=node.label,
                    start_line=node.evidence.start_line,
                    end_line=node.evidence.end_line,
                    text=node.evidence.quote,
                )
            )
    if not blocks:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if text.strip():
            end = min(len(lines), 120)
            blocks.append(CodeBlock(symbol=None, start_line=1, end_line=end, text="\n".join(lines[:end]).strip()))
    return blocks


def _extract_python(
    source: str,
    lines: list[str],
    file_node: CodeNode,
    module_node: CodeNode,
    nodes: list[CodeNode],
    edges: list[CodeEdge],
    seen_nodes: set[str],
    seen_edges: set[tuple[str, str, str]],
) -> None:
    symbol_nodes: dict[str, CodeNode] = {}
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        tree = None

    for line_number, line in enumerate(lines, start=1):
        match = PY_IMPORT_RE.match(line)
        if match:
            imported = match.group(1) or match.group(2)
            imported_node = _node(source, "Module", imported, imported, line_number, line_number, line.strip(), EXACT_STATIC, {"imported": True})
            _add_node(nodes, seen_nodes, imported_node)
            _add_edge(edges, seen_edges, module_node, imported_node, "IMPORTS")

    if tree is not None:
        for item in ast.walk(tree):
            if isinstance(item, ast.ClassDef):
                node = _node_for_span(source, "Class", item.name, item.name, lines, item.lineno, getattr(item, "end_lineno", item.lineno), EXACT_STATIC, {})
                symbol_nodes[item.name] = node
                _add_node(nodes, seen_nodes, node)
                _add_edge(edges, seen_edges, module_node, node, "DEFINES")
                for child in item.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method = _node_for_span(source, "Method", child.name, f"{item.name}.{child.name}", lines, child.lineno, getattr(child, "end_lineno", child.lineno), EXACT_STATIC, {"class": item.name})
                        symbol_nodes[child.name] = method
                        _add_node(nodes, seen_nodes, method)
                        _add_edge(edges, seen_edges, node, method, "DEFINES")
                        _python_route_from_decorators(source, lines, child, method, nodes, edges, seen_nodes, seen_edges)
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _is_method(tree, item):
                node = _node_for_span(source, "Function", item.name, item.name, lines, item.lineno, getattr(item, "end_lineno", item.lineno), EXACT_STATIC, {})
                symbol_nodes[item.name] = node
                _add_node(nodes, seen_nodes, node)
                _add_edge(edges, seen_edges, module_node, node, "DEFINES")
                _python_route_from_decorators(source, lines, item, node, nodes, edges, seen_nodes, seen_edges)
        return

    for line_number, line in enumerate(lines, start=1):
        class_match = PY_CLASS_RE.match(line)
        func_match = PY_FUNC_RE.match(line)
        if class_match:
            node = _node(source, "Class", class_match.group(1), class_match.group(1), line_number, line_number, line.strip(), EXACT_STATIC, {})
            _add_node(nodes, seen_nodes, node)
            _add_edge(edges, seen_edges, module_node, node, "DEFINES")
        elif func_match:
            node = _node(source, "Function", func_match.group(1), func_match.group(1), line_number, line_number, line.strip(), EXACT_STATIC, {})
            _add_node(nodes, seen_nodes, node)
            _add_edge(edges, seen_edges, module_node, node, "DEFINES")


def _python_route_from_decorators(
    source: str,
    lines: list[str],
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    handler: CodeNode,
    nodes: list[CodeNode],
    edges: list[CodeEdge],
    seen_nodes: set[str],
    seen_edges: set[tuple[str, str, str]],
) -> None:
    for decorator in func.decorator_list:
        if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
            continue
        method = decorator.func.attr.lower()
        if method not in HTTP_METHODS:
            continue
        if not decorator.args or not isinstance(decorator.args[0], ast.Constant) or not isinstance(decorator.args[0].value, str):
            continue
        route = decorator.args[0].value
        line = getattr(decorator, "lineno", func.lineno)
        endpoint = _node(source, "Endpoint", f"{method.upper()} {route}", route, line, line, _quote(lines, line, line), EXACT_STATIC, {"method": method.upper(), "path": route})
        _add_node(nodes, seen_nodes, endpoint)
        _add_edge(edges, seen_edges, handler, endpoint, "REFERENCES")
        _add_edge(edges, seen_edges, endpoint, handler, "ROUTE_TO_HANDLER")


def _extract_ts_js(
    source: str,
    lines: list[str],
    file_node: CodeNode,
    module_node: CodeNode,
    nodes: list[CodeNode],
    edges: list[CodeEdge],
    seen_nodes: set[str],
    seen_edges: set[tuple[str, str, str]],
) -> None:
    symbol_nodes: dict[str, CodeNode] = {}
    class_stack: list[tuple[str, CodeNode, int]] = []
    brace_depth = 0

    for line_number, line in enumerate(lines, start=1):
        for imported in [*_all_matches(TS_IMPORT_RE, line), *_all_matches(TS_REQUIRE_RE, line)]:
            imported_node = _node(source, "Module", imported, imported, line_number, line_number, line.strip(), EXACT_STATIC, {"imported": True})
            _add_node(nodes, seen_nodes, imported_node)
            _add_edge(edges, seen_edges, module_node, imported_node, "IMPORTS")

        class_match = TS_CLASS_RE.match(line)
        if class_match:
            name = class_match.group(1)
            node = _node(source, "Class", name, name, line_number, line_number, line.strip(), EXACT_STATIC, {})
            symbol_nodes[name] = node
            _add_node(nodes, seen_nodes, node)
            _add_edge(edges, seen_edges, module_node, node, "DEFINES")
            class_stack.append((name, node, brace_depth + line.count("{") - line.count("}")))

        func_match = TS_FUNCTION_RE.match(line) or TS_ARROW_RE.match(line)
        if func_match:
            name = func_match.group(1)
            node = _node(source, "Function", name, name, line_number, line_number, line.strip(), EXACT_STATIC, {})
            symbol_nodes[name] = node
            _add_node(nodes, seen_nodes, node)
            _add_edge(edges, seen_edges, module_node, node, "DEFINES")

        if class_stack and not func_match:
            method_match = TS_METHOD_RE.match(line)
            if method_match and method_match.group(1) not in {"if", "for", "while", "switch", "catch", "function"}:
                class_name, class_node, _ = class_stack[-1]
                name = method_match.group(1)
                node = _node(source, "Method", name, f"{class_name}.{name}", line_number, line_number, line.strip(), EXACT_STATIC, {"class": class_name})
                symbol_nodes[name] = node
                _add_node(nodes, seen_nodes, node)
                _add_edge(edges, seen_edges, class_node, node, "DEFINES")

        for route_match in EXPRESS_ROUTE_RE.finditer(line):
            method = route_match.group(1).upper()
            route = route_match.group(2)
            handler_name = route_match.group(3)
            endpoint = _node(source, "Endpoint", f"{method} {route}", route, line_number, line_number, line.strip(), EXACT_STATIC, {"method": method, "path": route})
            _add_node(nodes, seen_nodes, endpoint)
            _add_edge(edges, seen_edges, module_node, endpoint, "REFERENCES")
            if handler_name and handler_name in symbol_nodes:
                _add_edge(edges, seen_edges, endpoint, symbol_nodes[handler_name], "ROUTE_TO_HANDLER")

        brace_depth += line.count("{") - line.count("}")
        while class_stack and brace_depth < class_stack[-1][2]:
            class_stack.pop()


def _extract_java(
    source: str,
    lines: list[str],
    file_node: CodeNode,
    module_node: CodeNode,
    nodes: list[CodeNode],
    edges: list[CodeEdge],
    seen_nodes: set[str],
    seen_edges: set[tuple[str, str, str]],
) -> None:
    symbol_nodes: dict[str, CodeNode] = {}
    class_node: CodeNode | None = None
    pending_mapping: tuple[str, str, int] | None = None

    for line_number, line in enumerate(lines, start=1):
        import_match = JAVA_IMPORT_RE.match(line)
        if import_match:
            imported = import_match.group(1)
            imported_node = _node(source, "Module", imported, imported, line_number, line_number, line.strip(), EXACT_STATIC, {"imported": True})
            _add_node(nodes, seen_nodes, imported_node)
            _add_edge(edges, seen_edges, module_node, imported_node, "IMPORTS")

        mapping_match = JAVA_MAPPING_RE.search(line)
        if mapping_match:
            method = mapping_match.group(1).replace("Request", "").upper() or "ANY"
            route = mapping_match.group(2) or "/"
            pending_mapping = (method, route, line_number)

        class_match = JAVA_CLASS_RE.search(line)
        if class_match:
            name = class_match.group(1)
            class_node = _node(source, "Class", name, name, line_number, line_number, line.strip(), EXACT_STATIC, {})
            symbol_nodes[name] = class_node
            _add_node(nodes, seen_nodes, class_node)
            _add_edge(edges, seen_edges, module_node, class_node, "DEFINES")

        method_match = JAVA_METHOD_RE.match(line)
        if method_match and method_match.group(1) not in {"if", "for", "while", "switch", "catch"}:
            name = method_match.group(1)
            owner = class_node or module_node
            canonical = f"{owner.label}.{name}" if owner.kind == "Class" else name
            method_node = _node(source, "Method", name, canonical, line_number, line_number, line.strip(), EXACT_STATIC, {"class": owner.label if owner.kind == "Class" else None})
            symbol_nodes[name] = method_node
            _add_node(nodes, seen_nodes, method_node)
            _add_edge(edges, seen_edges, owner, method_node, "DEFINES")
            if pending_mapping:
                http_method, route, route_line = pending_mapping
                endpoint = _node(source, "Endpoint", f"{http_method} {route}", route, route_line, route_line, _quote(lines, route_line, route_line), EXACT_STATIC, {"method": http_method, "path": route})
                _add_node(nodes, seen_nodes, endpoint)
                _add_edge(edges, seen_edges, endpoint, method_node, "ROUTE_TO_HANDLER")
                pending_mapping = None


def _extract_sql(
    source: str,
    lines: list[str],
    file_node: CodeNode,
    nodes: list[CodeNode],
    edges: list[CodeEdge],
    seen_nodes: set[str],
    seen_edges: set[tuple[str, str, str]],
) -> None:
    text = "\n".join(lines)
    table_nodes: dict[str, CodeNode] = {}
    for match in SQL_TABLE_RE.finditer(text):
        table = _clean_identifier(match.group(1))
        line = _line_for_offset(text, match.start())
        relation = "DEFINES" if SQL_CREATE_RE.search(match.group(0)) else "QUERY_TOUCHES_TABLE"
        node = table_nodes.get(table) or _node(source, "Table", table, table, line, line, _quote(lines, line, line), EXACT_STATIC, {})
        table_nodes[table] = node
        _add_node(nodes, seen_nodes, node)
        _add_edge(edges, seen_edges, file_node, node, relation)

    for create_match in SQL_CREATE_RE.finditer(text):
        table = _clean_identifier(create_match.group(1))
        table_node = table_nodes.get(table)
        if not table_node:
            continue
        open_paren = text.find("(", create_match.end())
        start_line = _line_for_offset(text, open_paren) + 1 if open_paren != -1 else _line_for_offset(text, create_match.end()) + 1
        for line_number in range(start_line, len(lines) + 1):
            line = lines[line_number - 1]
            if ");" in line or line.strip().startswith(")"):
                break
            column_match = SQL_COLUMN_RE.match(line)
            if column_match:
                column = _clean_identifier(column_match.group(1))
                if column.upper() in {"CONSTRAINT", "PRIMARY", "FOREIGN", "UNIQUE", "CHECK"}:
                    continue
                node = _node(source, "Column", column, f"{table}.{column}", line_number, line_number, line.strip(), EXACT_STATIC, {"table": table})
                _add_node(nodes, seen_nodes, node)
                _add_edge(edges, seen_edges, table_node, node, "CONTAINS")


def _extract_openapi(
    source: str,
    text: str,
    lines: list[str],
    file_node: CodeNode,
    nodes: list[CodeNode],
    edges: list[CodeEdge],
    seen_nodes: set[str],
    seen_edges: set[tuple[str, str, str]],
) -> None:
    if source.endswith(".json"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            _json_config_nodes(source, data, lines, file_node, nodes, edges, seen_nodes, seen_edges)
            paths = data.get("paths", {})
            if isinstance(paths, dict):
                for route, methods in paths.items():
                    if isinstance(methods, dict):
                        for method in methods:
                            if method.lower() in HTTP_METHODS:
                                line = _find_line(lines, route)
                                endpoint = _node(source, "Endpoint", f"{method.upper()} {route}", route, line, line, _quote(lines, line, line), EXACT_STATIC, {"method": method.upper(), "path": route})
                                _add_node(nodes, seen_nodes, endpoint)
                                _add_edge(edges, seen_edges, file_node, endpoint, "DEFINES")
            return

    current_path: tuple[str, int] | None = None
    for line_number, line in enumerate(lines, start=1):
        match = YAML_KEY_RE.match(line)
        if match:
            key = match.group(2).strip("'\"")
            if key.startswith("/"):
                current_path = (key, line_number)
            node = _node(source, "ConfigKey", key, key, line_number, line_number, line.strip(), EXACT_STATIC, {})
            _add_node(nodes, seen_nodes, node)
            _add_edge(edges, seen_edges, file_node, node, "CONTAINS")
        method_match = YAML_METHOD_RE.match(line)
        if method_match and current_path:
            method = method_match.group(1).upper()
            route, route_line = current_path
            endpoint = _node(source, "Endpoint", f"{method} {route}", route, line_number, line_number, line.strip(), EXACT_STATIC, {"method": method, "path": route})
            _add_node(nodes, seen_nodes, endpoint)
            _add_edge(edges, seen_edges, file_node, endpoint, "DEFINES")


def _extract_config(
    source: str,
    text: str,
    lines: list[str],
    file_node: CodeNode,
    nodes: list[CodeNode],
    edges: list[CodeEdge],
    seen_nodes: set[str],
    seen_edges: set[tuple[str, str, str]],
) -> None:
    if source.endswith(".json"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            _json_config_nodes(source, data, lines, file_node, nodes, edges, seen_nodes, seen_edges)
            return
    for line_number, line in enumerate(lines, start=1):
        match = YAML_KEY_RE.match(line)
        if match:
            key = match.group(2).strip("'\"")
            node = _node(source, "ConfigKey", key, key, line_number, line_number, line.strip(), EXACT_STATIC, {})
            _add_node(nodes, seen_nodes, node)
            _add_edge(edges, seen_edges, file_node, node, "CONTAINS")


def _json_config_nodes(
    source: str,
    data: dict[str, Any],
    lines: list[str],
    file_node: CodeNode,
    nodes: list[CodeNode],
    edges: list[CodeEdge],
    seen_nodes: set[str],
    seen_edges: set[tuple[str, str, str]],
    prefix: str = "",
) -> None:
    for key, value in data.items():
        canonical = f"{prefix}.{key}" if prefix else str(key)
        line = _find_line(lines, f'"{key}"')
        node = _node(source, "ConfigKey", str(key), canonical, line, line, _quote(lines, line, line), EXACT_STATIC, {})
        _add_node(nodes, seen_nodes, node)
        _add_edge(edges, seen_edges, file_node, node, "CONTAINS")
        if isinstance(value, dict):
            _json_config_nodes(source, value, lines, file_node, nodes, edges, seen_nodes, seen_edges, canonical)


def _is_method(tree: ast.AST, func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and func in node.body:
            return True
    return False


def _node_for_span(
    source: str,
    kind: str,
    label: str,
    canonical: str,
    lines: list[str],
    start: int,
    end: int,
    confidence: str,
    metadata: dict[str, Any],
) -> CodeNode:
    return _node(source, kind, label, canonical, start, end, _quote(lines, start, min(end, start + 40)), confidence, metadata)


def _node(
    source: str,
    kind: str,
    label: str,
    canonical: str,
    start: int,
    end: int,
    quote: str,
    confidence: str,
    metadata: dict[str, Any],
) -> CodeNode:
    evidence = CodeEvidence(source, max(1, start), max(1, end), quote or source, confidence)
    return CodeNode(
        id=_node_id(source, kind, canonical, evidence.start_line, evidence.end_line, evidence.quote),
        kind=kind,
        label=label,
        canonical_name=canonical,
        evidence=evidence,
        metadata=metadata,
    )


def _add_node(nodes: list[CodeNode], seen: set[str], node: CodeNode) -> None:
    if node.id in seen:
        return
    seen.add(node.id)
    nodes.append(node)


def _add_edge(edges: list[CodeEdge], seen: set[tuple[str, str, str]], source: CodeNode, target: CodeNode, relation: str) -> None:
    key = (source.id, target.id, relation)
    if key in seen:
        return
    seen.add(key)
    edges.append(CodeEdge(source=source.id, target=target.id, relation=relation, evidence=target.evidence, confidence=target.evidence.confidence))


def _language(path: Path, kind: str | None) -> str:
    if kind in {"sql", "openapi", "config"}:
        return kind
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".ts", ".tsx"}:
        return "typescript"
    if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript"
    if suffix == ".java":
        return "java"
    if suffix == ".sql":
        return "sql"
    if path.name.lower() in {"openapi.json", "openapi.yaml", "openapi.yml", "swagger.json", "swagger.yaml", "swagger.yml"}:
        return "openapi"
    if suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"}:
        return "config"
    return "code"


def _module_name(source: str, language: str, lines: list[str]) -> str:
    if language == "java":
        for line in lines:
            match = JAVA_PACKAGE_RE.match(line)
            if match:
                return match.group(1)
    if language == "python":
        return source.rsplit(".", 1)[0].replace("/", ".")
    return source


def _quote(lines: list[str], start: int, end: int) -> str:
    if not lines:
        return ""
    start = max(1, start)
    end = min(len(lines), max(start, end))
    return "\n".join(lines[start - 1 : end]).strip()


def _all_matches(pattern: re.Pattern[str], line: str) -> list[str]:
    return [match.group(1) for match in pattern.finditer(line) if match.group(1)]


def _clean_identifier(value: str) -> str:
    return value.strip().strip('"`').split(".")[-1]


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _find_line(lines: list[str], needle: str) -> int:
    for index, line in enumerate(lines, start=1):
        if needle in line:
            return index
    return 1


def _node_id(source: str, kind: str, canonical: str, start: int, end: int, quote: str) -> str:
    value = "\n".join([source, kind, canonical, str(start), str(end), quote])
    return f"code:{sha256_text(value)[:24]}"
