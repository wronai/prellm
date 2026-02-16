"""CodebaseIndexer — tree-sitter based codebase indexing for context enrichment.

Parses source files using tree-sitter to extract functions, classes, imports,
and structural features. Provides a searchable index for preLLM context injection.

Requires: pip install tree-sitter tree-sitter-python tree-sitter-javascript
(or install prellm with extras when available)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("prellm.context.codebase_indexer")


@dataclass
class CodeSymbol:
    """A code symbol extracted from source."""
    name: str
    kind: str  # "function", "class", "method", "import", "variable"
    file_path: str
    line_start: int
    line_end: int
    signature: str = ""
    docstring: str = ""
    parent: str | None = None


@dataclass
class FileIndex:
    """Index of a single source file."""
    path: str
    language: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    loc: int = 0


@dataclass
class CodebaseIndex:
    """Full codebase index."""
    root: str
    files: list[FileIndex] = field(default_factory=list)
    total_symbols: int = 0
    total_files: int = 0
    total_loc: int = 0


# Language extension mapping
_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
}


class CodebaseIndexer:
    """Index a codebase using tree-sitter for AST-based symbol extraction.

    Usage:
        indexer = CodebaseIndexer()
        index = indexer.index_directory("/path/to/project")
        symbols = indexer.search(index, "deploy")
        context = indexer.get_context_for_query(index, "refactor the deploy function")
    """

    def __init__(self, languages: list[str] | None = None):
        """Initialize indexer.

        Args:
            languages: List of languages to index (e.g. ["python", "javascript"]).
                       If None, indexes all supported languages found.
        """
        self._languages = languages
        self._parsers: dict[str, Any] = {}
        self._tree_sitter_available = self._check_tree_sitter()

    @staticmethod
    def _check_tree_sitter() -> bool:
        """Check if tree-sitter is available."""
        try:
            import tree_sitter  # noqa: F401
            return True
        except ImportError:
            logger.info("tree-sitter not installed. Using fallback regex-based indexing.")
            return False

    def index_directory(
        self,
        root: str | Path,
        exclude_dirs: list[str] | None = None,
        max_file_size_kb: int = 500,
    ) -> CodebaseIndex:
        """Index all source files in a directory.

        Args:
            root: Root directory to index.
            exclude_dirs: Directory names to skip (default: common non-source dirs).
            max_file_size_kb: Skip files larger than this (KB).

        Returns:
            CodebaseIndex with all extracted symbols.
        """
        root = Path(root)
        exclude = set(exclude_dirs or [
            ".git", ".venv", "venv", "node_modules", "__pycache__",
            ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
        ])

        index = CodebaseIndex(root=str(root))

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in exclude for part in file_path.parts):
                continue
            if file_path.suffix not in _LANG_MAP:
                continue
            if file_path.stat().st_size > max_file_size_kb * 1024:
                continue

            lang = _LANG_MAP[file_path.suffix]
            if self._languages and lang not in self._languages:
                continue

            try:
                file_index = self._index_file(file_path, lang)
                index.files.append(file_index)
                index.total_symbols += len(file_index.symbols)
                index.total_loc += file_index.loc
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")

        index.total_files = len(index.files)
        logger.info(
            f"Indexed {index.total_files} files, "
            f"{index.total_symbols} symbols, "
            f"{index.total_loc} LOC"
        )
        return index

    def _index_file(self, file_path: Path, language: str) -> FileIndex:
        """Index a single file."""
        content = file_path.read_text(errors="replace")
        lines = content.splitlines()

        file_index = FileIndex(
            path=str(file_path),
            language=language,
            loc=len(lines),
        )

        if self._tree_sitter_available:
            file_index.symbols = self._extract_with_tree_sitter(content, language, str(file_path))
        else:
            file_index.symbols = self._extract_with_regex(content, language, str(file_path))

        # Extract imports (simple regex fallback always works)
        file_index.imports = self._extract_imports(content, language)

        return file_index

    def _extract_with_tree_sitter(self, content: str, language: str, file_path: str) -> list[CodeSymbol]:
        """Extract symbols using tree-sitter AST parsing."""
        try:
            parser = self._get_parser(language)
            if parser is None:
                return self._extract_with_regex(content, language, file_path)

            tree = parser.parse(bytes(content, "utf-8"))
            symbols: list[CodeSymbol] = []

            self._walk_tree(tree.root_node, symbols, file_path, content)
            return symbols

        except Exception as e:
            logger.debug(f"tree-sitter parsing failed for {file_path}: {e}")
            return self._extract_with_regex(content, language, file_path)

    def _get_parser(self, language: str) -> Any:
        """Get or create a tree-sitter parser for the given language."""
        if language in self._parsers:
            return self._parsers[language]

        try:
            import tree_sitter
            lang_module = __import__(f"tree_sitter_{language}")
            lang = tree_sitter.Language(lang_module.language())
            parser = tree_sitter.Parser(lang)
            self._parsers[language] = parser
            return parser
        except (ImportError, Exception) as e:
            logger.debug(f"No tree-sitter grammar for {language}: {e}")
            self._parsers[language] = None
            return None

    def _walk_tree(self, node: Any, symbols: list[CodeSymbol], file_path: str, content: str) -> None:
        """Recursively walk tree-sitter AST and extract symbols."""
        node_type = node.type

        if node_type in ("function_definition", "function_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                symbols.append(CodeSymbol(
                    name=name_node.text.decode("utf-8"),
                    kind="function",
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=self._get_line(content, node.start_point[0]),
                ))

        elif node_type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                symbols.append(CodeSymbol(
                    name=name_node.text.decode("utf-8"),
                    kind="class",
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=self._get_line(content, node.start_point[0]),
                ))

        for child in node.children:
            self._walk_tree(child, symbols, file_path, content)

    @staticmethod
    def _get_line(content: str, line_idx: int) -> str:
        """Get a specific line from content."""
        lines = content.splitlines()
        if 0 <= line_idx < len(lines):
            return lines[line_idx].strip()
        return ""

    @staticmethod
    def _extract_with_regex(content: str, language: str, file_path: str) -> list[CodeSymbol]:
        """Fallback: extract symbols using regex patterns."""
        import re
        symbols: list[CodeSymbol] = []
        lines = content.splitlines()

        if language == "python":
            for i, line in enumerate(lines, 1):
                # Functions
                m = re.match(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(", line)
                if m:
                    indent = len(m.group(1))
                    symbols.append(CodeSymbol(
                        name=m.group(2),
                        kind="method" if indent > 0 else "function",
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        signature=line.strip(),
                    ))
                # Classes
                m = re.match(r"^class\s+(\w+)", line)
                if m:
                    symbols.append(CodeSymbol(
                        name=m.group(1),
                        kind="class",
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        signature=line.strip(),
                    ))

        elif language in ("javascript", "typescript"):
            for i, line in enumerate(lines, 1):
                m = re.match(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", line)
                if m:
                    symbols.append(CodeSymbol(
                        name=m.group(1),
                        kind="function",
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        signature=line.strip(),
                    ))
                m = re.match(r"^\s*(?:export\s+)?class\s+(\w+)", line)
                if m:
                    symbols.append(CodeSymbol(
                        name=m.group(1),
                        kind="class",
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        signature=line.strip(),
                    ))

        return symbols

    @staticmethod
    def _extract_imports(content: str, language: str) -> list[str]:
        """Extract import statements."""
        import re
        imports: list[str] = []

        if language == "python":
            for line in content.splitlines():
                if re.match(r"^\s*(import|from)\s+", line):
                    imports.append(line.strip())
        elif language in ("javascript", "typescript"):
            for line in content.splitlines():
                if re.match(r"^\s*(import|require)\s*", line):
                    imports.append(line.strip())

        return imports

    def search(self, index: CodebaseIndex, query: str, limit: int = 20) -> list[CodeSymbol]:
        """Search the index for symbols matching a query string.

        Simple substring matching on symbol names and signatures.
        """
        query_lower = query.lower()
        matches: list[CodeSymbol] = []

        for file_index in index.files:
            for symbol in file_index.symbols:
                score = 0
                if query_lower in symbol.name.lower():
                    score += 10
                if query_lower in symbol.signature.lower():
                    score += 5
                if score > 0:
                    matches.append(symbol)

        return sorted(matches, key=lambda s: s.name.lower().find(query_lower))[:limit]

    def get_context_for_query(self, index: CodebaseIndex, query: str, max_symbols: int = 10) -> str:
        """Build a context string from the index for a given query.

        Returns a formatted string suitable for injection into LLM prompts.
        """
        symbols = self.search(index, query, limit=max_symbols)

        if not symbols:
            return f"[Codebase: {index.total_files} files, {index.total_symbols} symbols, no matches for '{query}']"

        lines = [f"[Codebase context: {index.total_files} files, {index.total_symbols} symbols]"]
        for s in symbols:
            lines.append(f"  {s.kind} {s.name} @ {s.file_path}:{s.line_start} — {s.signature}")

        return "\n".join(lines)

    def get_compressed_context(
        self, root: str | Path, query: str, max_tokens: int = 2048
    ) -> str:
        """Full pipeline: index → compress → filter by query relevance.

        Returns text ready for injection into small-LLM prompt.
        Guarantees output fits within max_tokens estimate.
        """
        from prellm.context.folder_compressor import FolderCompressor

        compressor = FolderCompressor(indexer=self)
        compressed = compressor.compress(root)

        # Start with summary (cheapest)
        parts: list[str] = []
        token_count = 0

        # Module summaries filtered by query relevance
        query_lower = query.lower()
        relevant_summaries: list[str] = []
        other_summaries: list[str] = []

        for mod, summary in compressed.module_summaries.items():
            line = f"  {mod}: {summary}"
            if any(w in mod.lower() or w in summary.lower() for w in query_lower.split()):
                relevant_summaries.append(line)
            else:
                other_summaries.append(line)

        header = f"[Project: {compressed.total_modules} modules, {compressed.total_functions} functions]"
        parts.append(header)
        token_count += self.estimate_tokens(header)

        # Add relevant summaries first
        for line in relevant_summaries + other_summaries:
            est = self.estimate_tokens(line)
            if token_count + est > max_tokens:
                break
            parts.append(line)
            token_count += est

        # Add dependency graph if space allows
        if compressed.dependency_graph and token_count < max_tokens * 0.8:
            dep_header = "\n[Dependencies]"
            parts.append(dep_header)
            token_count += self.estimate_tokens(dep_header)
            for mod, deps in compressed.dependency_graph.items():
                line = f"  {mod} → {', '.join(deps)}"
                est = self.estimate_tokens(line)
                if token_count + est > max_tokens:
                    break
                parts.append(line)
                token_count += est

        return "\n".join(parts)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (~4 chars/token) with margin."""
        return len(text) // 4 + 1
