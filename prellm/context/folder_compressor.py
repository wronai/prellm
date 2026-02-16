"""FolderCompressor — compresses project folder into lightweight representation for LLM context.

Extends CodebaseIndexer to provide:
- .toon format output (standardized)
- Dependency graph (internal imports)
- Module summaries (from docstrings/names)
- Token estimation
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from prellm.context.codebase_indexer import CodebaseIndex, CodebaseIndexer
from prellm.models import CompressedFolder

logger = logging.getLogger("prellm.context.folder_compressor")


class FolderCompressor:
    """Compresses a project folder into a lightweight representation for LLM context."""

    def __init__(self, indexer: CodebaseIndexer | None = None):
        self._indexer = indexer or CodebaseIndexer()

    def compress(
        self,
        root: str | Path,
        format: str = "toon",
        exclude_dirs: list[str] | None = None,
    ) -> CompressedFolder:
        """Compress folder to a lightweight representation.

        Args:
            root: Root directory to compress.
            format: Output format — "toon" (default).
            exclude_dirs: Directories to exclude.

        Returns:
            CompressedFolder with toon output, dependency graph, and summaries.
        """
        root = Path(root)
        index = self._indexer.index_directory(root, exclude_dirs=exclude_dirs)

        toon_output = self.to_toon(index)
        dep_graph = self.to_dependency_graph(index)
        summaries = self.to_summary(index)
        token_estimate = self.estimate_token_count(toon_output)

        total_functions = sum(
            len([s for s in f.symbols if s.kind in ("function", "method")])
            for f in index.files
        )

        return CompressedFolder(
            root=str(root),
            toon_output=toon_output,
            dependency_graph=dep_graph,
            module_summaries=summaries,
            total_modules=index.total_files,
            total_functions=total_functions,
            estimated_tokens=token_estimate,
        )

    def to_toon(self, index: CodebaseIndex) -> str:
        """Format index as .toon — standardized compact representation."""
        lines: list[str] = []
        project_name = Path(index.root).name
        lines.append(f"project: {project_name}")
        lines.append(f"generated: {datetime.utcnow().isoformat()}")
        lines.append(f"modules[{index.total_files}]{{path,lang,items}}:")

        for fi in index.files:
            rel = _relative_path(fi.path, index.root)
            lines.append(f"  {rel},{fi.language},{len(fi.symbols)}")

        lines.append("function_details:")
        for fi in index.files:
            rel = _relative_path(fi.path, index.root)
            funcs = [s for s in fi.symbols if s.kind in ("function", "method")]
            if not funcs:
                continue
            lines.append(f"  {rel}:")
            lines.append(f"    functions[{len(funcs)}]{{name,kind,sig,lines}}:")
            for s in funcs:
                sig = s.signature[:80] if s.signature else s.name
                line_count = max(1, s.line_end - s.line_start + 1)
                lines.append(f"      {s.name},{s.kind},{sig},{line_count}")

        return "\n".join(lines)

    def to_dependency_graph(self, index: CodebaseIndex) -> dict[str, list[str]]:
        """Build module → list of imported internal modules."""
        # Collect all module paths and top-level package names for internal detection
        project_name = Path(index.root).name
        internal_modules: set[str] = set()
        top_level_packages: set[str] = set()
        for fi in index.files:
            rel = _relative_path(fi.path, index.root)
            mod = _path_to_module(rel)
            if mod:
                internal_modules.add(mod)
                top_level_packages.add(mod.split(".")[0])

        # Try matching imports against root dir name AND all top-level packages
        candidate_names = {project_name} | top_level_packages

        graph: dict[str, list[str]] = {}
        for fi in index.files:
            rel = _relative_path(fi.path, index.root)
            mod = _path_to_module(rel)
            if not mod:
                continue

            deps: list[str] = []
            for imp in fi.imports:
                for name in candidate_names:
                    imported_mod = _extract_module_from_import(imp, name)
                    if imported_mod and imported_mod in internal_modules and imported_mod != mod:
                        deps.append(imported_mod)
                        break

            if deps:
                graph[mod] = sorted(set(deps))

        return graph

    def to_summary(self, index: CodebaseIndex) -> dict[str, str]:
        """Generate 1-line summary per module from docstrings or names."""
        summaries: dict[str, str] = {}

        for fi in index.files:
            rel = _relative_path(fi.path, index.root)
            mod = _path_to_module(rel)
            if not mod:
                continue

            # Try to extract module docstring from first few lines
            summary = _extract_module_docstring(fi.path)
            if not summary:
                # Fallback: generate from symbol names
                func_names = [s.name for s in fi.symbols if s.kind in ("function", "method")][:5]
                class_names = [s.name for s in fi.symbols if s.kind == "class"]
                parts = []
                if class_names:
                    parts.append(f"classes: {', '.join(class_names)}")
                if func_names:
                    parts.append(f"functions: {', '.join(func_names)}")
                summary = "; ".join(parts) if parts else f"{fi.language} module ({fi.loc} LOC)"

            summaries[mod] = summary[:120]

        return summaries

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """Estimate token count — rough approximation (1 token ≈ 4 chars)."""
        return len(text) // 4


def _relative_path(path: str, root: str) -> str:
    """Get relative path from root."""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def _path_to_module(rel_path: str) -> str:
    """Convert file path to Python module name."""
    if not rel_path.endswith(".py"):
        return rel_path.rsplit(".", 1)[0].replace("/", ".")
    mod = rel_path[:-3].replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[:-9]
    return mod


def _extract_module_from_import(import_line: str, project_name: str) -> str | None:
    """Extract module name from import statement, only return internal imports."""
    # from prellm.foo.bar import X
    m = re.match(r"from\s+([\w.]+)\s+import", import_line)
    if m:
        mod = m.group(1)
        if mod.startswith(project_name + ".") or mod == project_name:
            return mod
        return None

    # import prellm.foo.bar
    m = re.match(r"import\s+([\w.]+)", import_line)
    if m:
        mod = m.group(1)
        if mod.startswith(project_name + ".") or mod == project_name:
            return mod
    return None


def _extract_module_docstring(path: str) -> str:
    """Extract first line of module docstring."""
    try:
        with open(path, errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line)
                if i > 10:
                    break

        content = "".join(lines)
        # Match triple-quoted docstring at module level
        m = re.match(r'^(?:\s*#[^\n]*\n)*\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', content, re.DOTALL)
        if m:
            doc = m.group(1).strip()
            first_line = doc.split("\n")[0].strip()
            # Strip trailing " — " style suffix for brevity
            if " — " in first_line:
                first_line = first_line.split(" — ", 1)[0].strip()
            return first_line
    except Exception:
        pass
    return ""
