"""
HIVE — Self-Organizing Cognition for Code Architecture
═══════════════════════════════════════════════════════════════════════════════

Code IS meaning. Structure IS understanding.
No LLM required for architectural self-awareness.

Modules:
  hive_code_chunker.py  — AST-based code parsing
  code_schema_10k.py    — 10kD dimensions for code concepts
  cognition_graph.py    — Self-organizing architecture discovery

The key insight:
  1. Parse code with AST → structural chunks
  2. Encode chunks to 10kD using code schema
  3. Similarity matrix finds natural boundaries
  4. Architecture understands itself

Born: 2026-01-03
"""

__version__ = "0.1.0"

from .hive_code_chunker import (
    HiveCodeChunker,
    CodeChunk,
    ChunkType,
)

from .code_schema_10k import (
    CODE_SCHEMA_10K,
    CodeSchemaEncoder,
)

from .cognition_graph import (
    CognitionGraph,
    ArchitectureCluster,
)

__all__ = [
    # Chunker
    "HiveCodeChunker",
    "CodeChunk",
    "ChunkType",
    # Schema
    "CODE_SCHEMA_10K",
    "CodeSchemaEncoder",
    # Graph
    "CognitionGraph",
    "ArchitectureCluster",
]
