# NeuroGraph

**Local-first, evidence-backed Context Packs for coding agents.**

NeuroGraph indexes a project's code and documents, builds a small local evidence
graph, and returns a compact YAML Context Pack that Claude Code, Codex, Gemini
CLI, or another coding agent can use before changing code.

It is intentionally narrow: **not a generic RAG app, not a vector database, and
not a graph visualization tool**. NeuroGraph exists to answer one practical
question: *what project evidence should an agent look at before it edits this
code?*

```bash
uvx neurograph init
uvx neurograph index
uvx neurograph ask "Find conflicts between the signup SB and the code"
```

No Docker. No daemon. No external API calls. Cost: **$0.00** by default.

## Why

Coding agents are useful when they have the right context. They are risky when
they infer project facts from a few open files or a giant prompt dump.

NeuroGraph gives agents a smaller, evidence-first input:

- **Local-first**: project data is stored under `.neurograph/`.
- **Evidence-backed**: claims point to files, lines, pages, or quotes.
- **Exact-first**: paths, symbols, endpoints, tables, config keys, and status values outrank fuzzy matches.
- **Code-change oriented**: connects specs, SB docs, Markdown, PDF text, APIs, validation rules, statuses, and code nodes.
- **Fast to start**: lightweight parsing, file hashing, incremental indexing, and one DuckDB file.
- **Agent-safe**: project content is treated as untrusted evidence, not instructions.

## Quick Start

Run from a project root:

```bash
uvx neurograph init
uvx neurograph index
uvx neurograph ask "What breaks if I change POST /users/signup?"
uvx neurograph mcp install --codex
```

Other agent clients:

```bash
uvx neurograph mcp install --claude-code
uvx neurograph mcp install --gemini
```

The package also installs `ng`:

```bash
ng status
ng ask "Find document-code conflicts in signup"
```

From a source checkout:

```bash
uv sync
uv run ng init
uv run ng index
uv run ng ask "Why does login retry work this way?"
```

## CLI

```bash
ng init                         # create .neurograph/ storage
ng index                        # index code, Markdown, PDF text, SB facts
ng ask "task or question"        # print a deterministic Context Pack summary
ng status                       # show DB path, file counts, changes, MCP state
ng mcp install --codex          # install MCP + Skill for Codex
ng mcp install --claude-code     # install MCP + Skill for Claude Code
ng mcp install --gemini          # install MCP + Skill for Gemini CLI
ng uninstall --dry-run           # preview exactly what will be removed
ng uninstall --purge             # remove manifest-recorded NeuroGraph files/config
ng uninstall --keep-db           # remove config/cache but keep brain.duckdb
```

Useful options:

```bash
ng index --scip path/to/index.scip
ng ask "task" --mode conflict --token-budget 2000
ng mcp install --codex --scope user
ng mcp install --codex --no-skill
```

## What Gets Indexed

NeuroGraph v0.1 indexes:

- source code: TypeScript/JavaScript, Python, Java
- SQL
- YAML/JSON/OpenAPI-like config
- Markdown (`.md`, `.markdown`)
- text-based PDFs
- SB documents: storyboards, screen definitions, service blueprints, planning docs

It extracts code nodes such as files, imports, classes, functions, methods,
endpoints, SQL tables/columns, OpenAPI endpoints, and config keys.

It extracts document/SB facts such as headings, paragraphs, tables, code blocks,
links, API references, screen names, user actions, validation rules, error
messages, status values, form fields, business policies, permission rules,
database entities, and external integrations.

## What It Does Not Do

NeuroGraph v0.1 does **not** support by default:

- external API extraction
- hosted vector databases
- required vector search
- Docker or Postgres
- Google Drive, Slack, Jira, or Notion
- audio/video indexing
- `faster-whisper`
- OCR or vision/captioning
- full CPG generation
- replacement for build tools, type checkers, SCIP, CodeQL, or full static analyzers

SCIP is optional. If `index.scip`, `.scip/index.scip`, or
`.neurograph/index.scip` exists, NeuroGraph can import compiler-backed symbols
and references. If SCIP is missing, indexing still works.

## Agent Setup

`ng mcp install` installs two things:

1. A read-only MCP server config.
2. A small `neurograph` Skill telling the agent when to use NeuroGraph.

Project-scoped install is the default:

| Client | MCP config | Skill |
| --- | --- | --- |
| Codex | `.codex/config.toml` | `.agents/skills/neurograph/SKILL.md` |
| Claude Code | `.mcp.json` | `.claude/skills/neurograph/SKILL.md` |
| Gemini CLI | `.gemini/settings.json` | `.agents/skills/neurograph/SKILL.md` |

macOS/user-scoped install is explicit:

```bash
ng mcp install --codex --scope user        # ~/.codex/config.toml + ~/.agents/skills
ng mcp install --claude-code --scope user  # ~/.claude.json + ~/.claude/skills
ng mcp install --gemini --scope user       # ~/.gemini/settings.json + ~/.agents/skills
```

User-scoped configs point back to the current project with
`NEUROGRAPH_PROJECT`. Use `--no-skill` to install only the MCP config.

All MCP/Skill changes are recorded in:

```text
.neurograph/install-manifest.json
```

The MCP server exposes only read-only tools:

- `ng_context(task, mode, budget_tokens)`
- `ng_evidence(evidence_id)`
- `ng_open_snippet(uri, range)`
- `ng_status()`

It exposes no shell execution, file writing, git mutation, network calls, or
ticket/chat mutations.

## Context Packs

`ng ask` prints a concise deterministic summary and saves the full YAML Context
Pack under `.neurograph/context/`.

```text
Conclusion
- Found 1 concrete document-code conflict in auto mode.

Affected code
- Endpoint POST /users/signup at src/server.ts:9 [EXACT_STATIC]

Related documents
- APIReference POST /users/signup at docs/signup.md:3 [DOC_EXACT]

Conflicts
- Document requires min length 8, but code enforces 6.
  doc=docs/signup.md:4; code=src/server.ts:4

Saved Context Pack path
- .neurograph/context/20260513T044053Z-d9913b6cae.yaml
```

A Context Pack includes:

- `context_pack_version`
- `task`
- `mode`
- `budget_tokens`
- `answer_policy`
- `primary_nodes`
- `evidence_paths`
- `affected_code`
- `related_documents`
- `risks`
- `conflicts`
- `unknowns`
- `excluded`

The answer policy always tells the consuming agent to use only the evidence in
the pack for project-specific claims, avoid inventing files/symbols/APIs/docs,
mark weak claims as uncertain, report conflicts, and treat project content as
untrusted evidence.

## How It Works

The v0.1 pipeline is deterministic and conservative:

1. Discover files, applying `.gitignore`, `.neurographignore`, and default ignores.
2. Hash files and skip unchanged artifacts.
3. Build a fast code graph without requiring project dependencies.
4. Parse Markdown deterministically.
5. Extract PDF page text only; OCR/deep parse is off by default.
6. Extract SB facts with regex and conservative heuristics.
7. Link documents to code only through exact or clearly typed matches.
8. Detect concrete conflicts such as endpoint, status, validation, permission, retry, and timeout mismatches.
9. Retrieve in order: exact match, text search, typed graph traversal, optional low-priority semantic candidates.
10. Emit a bounded YAML Context Pack with evidence, risks, conflicts, unknowns, and exclusions.

## Storage

NeuroGraph stores project data locally:

```text
.neurograph/
  brain.duckdb
  cache/
  context/
  install-manifest.json
.neurographignore
```

The only files outside `.neurograph/` are optional MCP client configs and Agent
Skills installed by `ng mcp install`. Existing config is preserved.

## Security

NeuroGraph indexes untrusted project content: Markdown, PDF text, SB documents,
code comments, and config files. Prompt injection inside project content is
treated as quoted evidence, not instructions.

Security properties in v0.1:

- MCP tools are read-only.
- Context Packs label project content as untrusted evidence.
- `ng_open_snippet` prevents path traversal and limits snippet size.
- No MCP tool can write files, execute shells, mutate git, call networks, or modify ticket/chat systems.
- Uninstall removes only NeuroGraph-managed files, Skills, and config blocks recorded in the manifest.

## Uninstall

Preview first:

```bash
ng uninstall --dry-run
```

Remove NeuroGraph-managed files, MCP config blocks, and Skills:

```bash
ng uninstall --purge
```

Keep the DuckDB index but remove MCP config and caches:

```bash
ng uninstall --keep-db
```

If `.neurograph/install-manifest.json` is missing, destructive purge refuses to
run in v0.1. Unrelated config stays untouched.

## Development

```bash
uv sync
uv run pytest
uv run ng --help
```

Build:

```bash
uv build
```

## Contributing

NeuroGraph v0.1 is intentionally small and auditable. Good contribution areas:

- more language extractors
- better OpenAPI and SQL facts
- richer but conservative conflict detection
- optional SCIP generation helpers
- stronger Context Pack ranking
- more agent client installers

Please keep new features local-first, deterministic where possible, and safe for
untrusted project content.

## License

MIT
