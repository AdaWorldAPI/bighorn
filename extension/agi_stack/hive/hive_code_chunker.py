#!/usr/bin/env python3
"""
hive_code_chunker.py — AST-based Code Chunking (No LLM Required)
═══════════════════════════════════════════════════════════════════════════════

Code IS already structured. This module extracts that structure.

What we extract:
  - Classes → ENTITIES (what exists)
  - Methods → CAPABILITIES (what can it do)
  - Imports → DEPENDENCIES (what does it need)
  - Calls → EDGES (who calls whom)
  - Docstrings → INTENT (what does it want)

The key insight:
  AST parsing gives us semantic chunks WITHOUT needing an LLM.
  The structure IS the meaning.

Born: 2026-01-03
"""

from __future__ import annotations
import ast
import os
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from enum import Enum


class ChunkType(Enum):
    """Types of code chunks we can extract."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    IMPORT = "import"
    CONSTANT = "constant"


@dataclass
class CodeChunk:
    """
    A semantic chunk of code extracted via AST.

    This is the atomic unit for self-organizing architecture.
    """
    # Identity
    id: str                              # Unique ID (file:type:name)
    chunk_type: ChunkType
    name: str
    file_path: str
    line_start: int
    line_end: int

    # Content
    docstring: Optional[str] = None
    code_preview: str = ""               # First 200 chars

    # Structure
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    args: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    decorators: List[str] = field(default_factory=list)

    # Relationships
    parent: Optional[str] = None         # Class this method belongs to
    inherits: List[str] = field(default_factory=list)
    contains: List[str] = field(default_factory=list)  # Methods in class

    # Fingerprint (for deduplication)
    fingerprint: str = ""

    # 10kD vector (filled later by encoder)
    vector: Optional[Any] = None

    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Compute structural fingerprint for similarity detection."""
        parts = [
            self.chunk_type.value,
            self.name,
            ",".join(sorted(self.imports)),
            ",".join(sorted(self.calls)),
            ",".join(self.args),
            self.returns or "",
        ]
        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.chunk_type.value,
            "name": self.name,
            "file": self.file_path,
            "lines": [self.line_start, self.line_end],
            "docstring": self.docstring,
            "imports": self.imports,
            "calls": self.calls,
            "args": self.args,
            "returns": self.returns,
            "decorators": self.decorators,
            "parent": self.parent,
            "inherits": self.inherits,
            "contains": self.contains,
            "fingerprint": self.fingerprint,
        }


class CallExtractor(ast.NodeVisitor):
    """Extract function/method calls from AST."""

    def __init__(self):
        self.calls: Set[str] = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # obj.method() → extract method name
            self.calls.add(node.func.attr)
            # Also track the object if it's a name
            if isinstance(node.func.value, ast.Name):
                self.calls.add(f"{node.func.value.id}.{node.func.attr}")
        self.generic_visit(node)


class HiveCodeChunker:
    """
    AST-based code chunker for self-organizing architecture.

    No LLM required. Code structure IS meaning.
    """

    def __init__(self):
        self.chunks: List[CodeChunk] = []
        self.file_chunks: Dict[str, List[CodeChunk]] = {}
        self.call_graph: Dict[str, Set[str]] = {}
        self.import_graph: Dict[str, Set[str]] = {}

    def chunk_file(self, filepath: str) -> List[CodeChunk]:
        """
        Parse a Python file and extract all chunks.

        Returns list of CodeChunk objects.
        """
        filepath = str(filepath)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            print(f"⚠ Could not read {filepath}: {e}")
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"⚠ Syntax error in {filepath}: {e}")
            return []

        chunks = []

        # Extract module-level imports
        module_imports = self._extract_imports(tree)

        # Create module chunk
        module_chunk = CodeChunk(
            id=f"{filepath}:module:__main__",
            chunk_type=ChunkType.MODULE,
            name=Path(filepath).stem,
            file_path=filepath,
            line_start=1,
            line_end=len(source.splitlines()),
            docstring=ast.get_docstring(tree),
            imports=module_imports,
            code_preview=source[:200],
        )
        chunks.append(module_chunk)

        # Walk AST for classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                chunk = self._extract_class(node, filepath, source)
                chunks.append(chunk)

                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                        method_chunk = self._extract_function(
                            item, filepath, source,
                            parent_class=node.name
                        )
                        chunks.append(method_chunk)
                        chunk.contains.append(method_chunk.id)

            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Only top-level functions (not methods)
                if not self._is_method(node, tree):
                    chunk = self._extract_function(node, filepath, source)
                    chunks.append(chunk)

        # Store
        self.chunks.extend(chunks)
        self.file_chunks[filepath] = chunks

        return chunks

    def chunk_directory(self, directory: str, pattern: str = "**/*.py") -> List[CodeChunk]:
        """
        Recursively chunk all Python files in a directory.
        """
        directory = Path(directory)
        all_chunks = []

        for filepath in directory.glob(pattern):
            # Skip __pycache__, .git, etc.
            if any(part.startswith('.') or part == '__pycache__' for part in filepath.parts):
                continue

            chunks = self.chunk_file(str(filepath))
            all_chunks.extend(chunks)
            print(f"  ✓ {filepath.name}: {len(chunks)} chunks")

        return all_chunks

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        return imports

    def _extract_class(self, node: ast.ClassDef, filepath: str, source: str) -> CodeChunk:
        """Extract a class definition."""
        # Get base classes
        inherits = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                inherits.append(base.id)
            elif isinstance(base, ast.Attribute):
                inherits.append(f"{base.value.id if isinstance(base.value, ast.Name) else '?'}.{base.attr}")

        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                decorators.append(dec.func.id)

        # Get method names
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)

        return CodeChunk(
            id=f"{filepath}:class:{node.name}",
            chunk_type=ChunkType.CLASS,
            name=node.name,
            file_path=filepath,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            decorators=decorators,
            inherits=inherits,
            contains=[f"{filepath}:method:{node.name}.{m}" for m in methods],
        )

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        filepath: str,
        source: str,
        parent_class: Optional[str] = None
    ) -> CodeChunk:
        """Extract a function or method."""
        # Get arguments
        args = [arg.arg for arg in node.args.args]

        # Get return type annotation
        returns = None
        if node.returns:
            returns = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                decorators.append(dec.func.id)

        # Extract calls
        call_extractor = CallExtractor()
        call_extractor.visit(node)
        calls = list(call_extractor.calls)

        # Determine type and name
        if parent_class:
            chunk_type = ChunkType.METHOD
            full_name = f"{parent_class}.{node.name}"
            chunk_id = f"{filepath}:method:{full_name}"
        else:
            chunk_type = ChunkType.FUNCTION
            full_name = node.name
            chunk_id = f"{filepath}:function:{node.name}"

        return CodeChunk(
            id=chunk_id,
            chunk_type=chunk_type,
            name=full_name,
            file_path=filepath,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            args=args,
            returns=returns,
            decorators=decorators,
            calls=calls,
            parent=parent_class,
        )

    def _is_method(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method (inside a class)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item is func_node:
                        return True
        return False

    def build_call_graph(self) -> Dict[str, Set[str]]:
        """
        Build call graph from all chunks.

        Returns: {caller_id: {callee_ids...}}
        """
        # First, build name → chunk_id mapping
        name_to_id: Dict[str, str] = {}
        for chunk in self.chunks:
            if chunk.chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD):
                # Map both short and full names
                short_name = chunk.name.split('.')[-1]
                name_to_id[short_name] = chunk.id
                name_to_id[chunk.name] = chunk.id

        # Build graph
        graph: Dict[str, Set[str]] = {}
        for chunk in self.chunks:
            if chunk.calls:
                callees = set()
                for call in chunk.calls:
                    if call in name_to_id:
                        callees.add(name_to_id[call])
                if callees:
                    graph[chunk.id] = callees

        self.call_graph = graph
        return graph

    def build_import_graph(self) -> Dict[str, Set[str]]:
        """
        Build import dependency graph.

        Returns: {file: {imported_modules...}}
        """
        graph: Dict[str, Set[str]] = {}

        for filepath, chunks in self.file_chunks.items():
            module_chunk = next((c for c in chunks if c.chunk_type == ChunkType.MODULE), None)
            if module_chunk and module_chunk.imports:
                graph[filepath] = set(module_chunk.imports)

        self.import_graph = graph
        return graph

    def find_duplicates(self, threshold: float = 0.9) -> List[Tuple[CodeChunk, CodeChunk, float]]:
        """
        Find potentially duplicate chunks by fingerprint similarity.

        Note: This is structural similarity, not semantic.
        Chunks with same fingerprint are likely duplicates.
        """
        from collections import defaultdict

        # Group by fingerprint
        by_fingerprint: Dict[str, List[CodeChunk]] = defaultdict(list)
        for chunk in self.chunks:
            if chunk.chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD):
                by_fingerprint[chunk.fingerprint].append(chunk)

        # Find groups with multiple chunks (exact duplicates)
        duplicates = []
        for fp, chunks in by_fingerprint.items():
            if len(chunks) > 1:
                for i, c1 in enumerate(chunks):
                    for c2 in chunks[i+1:]:
                        duplicates.append((c1, c2, 1.0))

        return duplicates

    def find_orphans(self) -> List[CodeChunk]:
        """
        Find chunks that are never called by anyone.

        Orphans are deletion candidates.
        """
        # Build set of all callees
        all_called = set()
        for callees in self.call_graph.values():
            all_called.update(callees)

        # Find chunks not in callees (excluding modules and classes)
        orphans = []
        for chunk in self.chunks:
            if chunk.chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD):
                if chunk.id not in all_called:
                    # Exclude __init__, __str__, etc.
                    if not chunk.name.split('.')[-1].startswith('__'):
                        orphans.append(chunk)

        return orphans

    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics."""
        by_type = {}
        for chunk in self.chunks:
            t = chunk.chunk_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_chunks": len(self.chunks),
            "files": len(self.file_chunks),
            "by_type": by_type,
            "call_graph_edges": sum(len(v) for v in self.call_graph.values()),
            "import_graph_edges": sum(len(v) for v in self.import_graph.values()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_chunker():
    """Test the code chunker."""
    print("=" * 60)
    print("HIVE CODE CHUNKER TEST")
    print("=" * 60)

    chunker = HiveCodeChunker()

    # Chunk this file
    chunks = chunker.chunk_file(__file__)

    print(f"\nChunked {__file__}:")
    print(f"  Total chunks: {len(chunks)}")

    for chunk in chunks[:5]:
        print(f"\n  [{chunk.chunk_type.value}] {chunk.name}")
        if chunk.docstring:
            print(f"      doc: {chunk.docstring[:50]}...")
        if chunk.calls:
            print(f"      calls: {chunk.calls[:5]}")
        if chunk.imports:
            print(f"      imports: {chunk.imports[:5]}")

    # Build graphs
    call_graph = chunker.build_call_graph()
    import_graph = chunker.build_import_graph()

    print(f"\nCall graph: {len(call_graph)} callers")
    print(f"Import graph: {len(import_graph)} files")

    # Find orphans
    orphans = chunker.find_orphans()
    print(f"\nOrphans (never called): {len(orphans)}")
    for o in orphans[:3]:
        print(f"  - {o.name}")

    # Stats
    stats = chunker.get_stats()
    print(f"\nStats: {stats}")

    print("\n" + "=" * 60)
    print("CHUNKER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_chunker()
