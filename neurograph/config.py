"""Project-wide configuration constants."""

from __future__ import annotations

APP_DIR_NAME = ".neurograph"
CACHE_DIR_NAME = "cache"
CONTEXT_DIR_NAME = "context"
MANIFEST_NAME = "install-manifest.json"
DB_NAME = "brain.duckdb"
IGNORE_FILE_NAME = ".neurographignore"

EXTERNAL_API_ENABLED = False

SUPPORTED_CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".clj",
    ".cpp",
    ".cs",
    ".dart",
    ".ex",
    ".exs",
    ".fish",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".lua",
    ".m",
    ".mm",
    ".pl",
    ".php",
    ".py",
    ".r",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
    ".vue",
    ".zsh",
}

SUPPORTED_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_SB_EXTENSIONS = {".sb", ".sourcebook"}
SUPPORTED_SQL_EXTENSIONS = {".sql"}
SUPPORTED_CONFIG_EXTENSIONS = {
    ".cfg",
    ".conf",
    ".env",
    ".gql",
    ".graphql",
    ".ini",
    ".json",
    ".toml",
    ".xml",
    ".yaml",
    ".yml",
}
OPENAPI_FILENAMES = {
    "openapi.json",
    "openapi.yaml",
    "openapi.yml",
    "swagger.json",
    "swagger.yaml",
    "swagger.yml",
}

DEFAULT_IGNORED_DIRS = [
    ".git",
    ".hg",
    ".neurograph",
    ".next",
    ".pytest_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
    "venv",
]

DEFAULT_IGNORED_PATTERNS = [
    ".DS_Store",
    "*.gif",
    "*.generated.*",
    "*.jpeg",
    "*.jpg",
    "*.lock",
    "*.min.js",
    "*.mov",
    "*.mp3",
    "*.mp4",
    "*.png",
    "*.snapshot",
    "*.wav",
    "*.webp",
]

DEFAULT_IGNORE_PATTERNS = [
    *(f"{directory}/" for directory in DEFAULT_IGNORED_DIRS),
    *DEFAULT_IGNORED_PATTERNS,
]
