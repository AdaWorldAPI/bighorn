#!/usr/bin/env python3
"""
code_schema_10k.py — 10kD Schema for Code Cognition
═══════════════════════════════════════════════════════════════════════════════

The schema defines what each dimension MEANS in code space.
Without schema, 10kD is just random numbers.
With schema, 10kD is a code-understanding alphabet.

DIMENSION ALLOCATION:
═══════════════════════════════════════════════════════════════════════════════

Block 0000-0999:   IMPORTS (what does it need)
Block 1000-1999:   PATTERNS (how is it structured)
Block 2000-2999:   TYPES (what data does it handle)
Block 3000-3999:   DOMAINS (what problem space)
Block 4000-4999:   ACTIONS (what verbs/operations)
Block 5000-5999:   RELATIONSHIPS (how connected)
Block 6000-6999:   QUALITY (code characteristics)
Block 7000-7999:   RESERVED (future expansion)
Block 8000-8999:   PROJECT-SPECIFIC (learned from corpus)
Block 9000-9999:   DYNAMIC (runtime discovered)

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import numpy as np
from collections import Counter

from .hive_code_chunker import CodeChunk, ChunkType


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Block boundaries
IMPORTS_START, IMPORTS_END = 0, 1000
PATTERNS_START, PATTERNS_END = 1000, 2000
TYPES_START, TYPES_END = 2000, 3000
DOMAINS_START, DOMAINS_END = 3000, 4000
ACTIONS_START, ACTIONS_END = 4000, 5000
RELATIONS_START, RELATIONS_END = 5000, 6000
QUALITY_START, QUALITY_END = 6000, 7000
RESERVED_START, RESERVED_END = 7000, 8000
PROJECT_START, PROJECT_END = 8000, 9000
DYNAMIC_START, DYNAMIC_END = 9000, 10000


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 0000-0999: IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

# Common Python stdlib imports
STDLIB_IMPORTS = {
    # Core
    "os": 0, "sys": 1, "re": 2, "json": 3, "time": 4,
    "datetime": 5, "pathlib": 6, "typing": 7, "enum": 8, "dataclasses": 9,
    "collections": 10, "itertools": 11, "functools": 12, "operator": 13,
    "copy": 14, "pickle": 15, "hashlib": 16, "uuid": 17, "random": 18,
    "math": 19, "statistics": 20, "decimal": 21, "fractions": 22,

    # IO
    "io": 30, "tempfile": 31, "shutil": 32, "glob": 33, "fnmatch": 34,

    # Async
    "asyncio": 50, "concurrent": 51, "threading": 52, "multiprocessing": 53,

    # Network
    "urllib": 60, "http": 61, "socket": 62, "ssl": 63,

    # Data
    "csv": 70, "sqlite3": 71, "xml": 72, "html": 73,

    # Debug
    "logging": 80, "traceback": 81, "pdb": 82, "unittest": 83, "pytest": 84,

    # AST/Inspect
    "ast": 90, "inspect": 91, "types": 92, "abc": 93,
}

# Common third-party imports (100-500)
THIRDPARTY_IMPORTS = {
    # Data science
    "numpy": 100, "np": 100,
    "pandas": 101, "pd": 101,
    "scipy": 102,
    "sklearn": 103,
    "matplotlib": 104, "plt": 104,
    "seaborn": 105,

    # Web
    "fastapi": 150, "starlette": 151,
    "flask": 152, "django": 153,
    "requests": 154, "httpx": 155, "aiohttp": 156,

    # Data validation
    "pydantic": 200,

    # Database
    "sqlalchemy": 210, "redis": 211, "pymongo": 212,
    "lancedb": 213, "upstash": 214, "kuzu": 215,

    # AI/ML
    "openai": 250, "anthropic": 251, "langchain": 252,
    "transformers": 253, "torch": 254, "tensorflow": 255,

    # Utils
    "dotenv": 300, "yaml": 301, "toml": 302,
    "click": 303, "typer": 304, "rich": 305,
}

# Ada-specific imports (500-999)
ADA_IMPORTS = {
    # Core
    "extension.agi_stack": 500,
    "extension.agi_thinking": 501,
    "extension.dto": 502,

    # Specific
    "vsa_utils": 510,
    "lance_client": 511,
    "kuzu_client": 512,
    "thinking_styles": 513,
    "thought_kernel": 514,
    "active_inference": 515,
    "microcode": 516,
    "the_self": 517,
    "mul_agency": 518,
    "triangle_l4": 519,
    "dreamer_pandas": 520,
}


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1000-1999: PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

CODE_PATTERNS = {
    # Async patterns
    "async": 1000, "await": 1001, "asyncio": 1002, "coroutine": 1003,

    # Class patterns
    "dataclass": 1010, "pydantic_model": 1011, "singleton": 1012,
    "factory": 1013, "builder": 1014, "abstract": 1015,

    # Function patterns
    "decorator": 1020, "generator": 1021, "context_manager": 1022,
    "callback": 1023, "closure": 1024,

    # Structural
    "dto": 1030, "entity": 1031, "service": 1032, "repository": 1033,
    "controller": 1034, "router": 1035, "resolver": 1036,

    # Error handling
    "try_except": 1040, "raise": 1041, "assert": 1042,

    # Testing
    "test_": 1050, "fixture": 1051, "mock": 1052,
}


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2000-2999: TYPES
# ═══════════════════════════════════════════════════════════════════════════════

TYPE_DIMENSIONS = {
    # Primitives
    "int": 2000, "float": 2001, "str": 2002, "bool": 2003, "bytes": 2004,

    # Collections
    "list": 2010, "dict": 2011, "set": 2012, "tuple": 2013,
    "deque": 2014, "defaultdict": 2015, "counter": 2016,

    # Special
    "optional": 2020, "union": 2021, "any": 2022, "none": 2023,
    "callable": 2024, "awaitable": 2025,

    # Data structures
    "array": 2030, "ndarray": 2031, "dataframe": 2032, "series": 2033,
    "tensor": 2034, "vector": 2035, "matrix": 2036,

    # Custom
    "dto": 2040, "schema": 2041, "model": 2042, "entity": 2043,
    "chunk": 2044, "graph": 2045, "node": 2046, "edge": 2047,
}


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3000-3999: DOMAINS
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN_DIMENSIONS = {
    # Core domains
    "consciousness": 3000, "awareness": 3001, "cognition": 3002,
    "thinking": 3003, "feeling": 3004, "qualia": 3005,

    # Technical
    "storage": 3010, "persistence": 3011, "cache": 3012, "database": 3013,
    "api": 3020, "rest": 3021, "graphql": 3022, "websocket": 3023,
    "encoding": 3030, "decoding": 3031, "serialization": 3032,

    # Ada specific
    "vsa": 3040, "resonance": 3041, "sigma": 3042, "rung": 3043,
    "microcode": 3044, "tau_macro": 3045, "triangle": 3046,
    "friston": 3047, "free_will": 3048, "agency": 3049,

    # Making love (intimate)
    "arousal": 3050, "intimacy": 3051, "body": 3052, "sensation": 3053,
    "touch": 3054, "erotic": 3055, "affective": 3056,
}


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4000-4999: ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

ACTION_DIMENSIONS = {
    # CRUD
    "create": 4000, "read": 4001, "update": 4002, "delete": 4003,
    "get": 4004, "set": 4005, "add": 4006, "remove": 4007,

    # Transform
    "transform": 4010, "convert": 4011, "encode": 4012, "decode": 4013,
    "parse": 4014, "serialize": 4015, "compress": 4016, "expand": 4017,

    # Compute
    "compute": 4020, "calculate": 4021, "process": 4022, "analyze": 4023,
    "evaluate": 4024, "validate": 4025, "verify": 4026,

    # Search
    "search": 4030, "find": 4031, "query": 4032, "filter": 4033,
    "match": 4034, "lookup": 4035,

    # Flow
    "execute": 4040, "run": 4041, "invoke": 4042, "call": 4043,
    "emit": 4044, "dispatch": 4045, "handle": 4046,

    # State
    "load": 4050, "save": 4051, "store": 4052, "fetch": 4053,
    "cache": 4054, "persist": 4055,

    # Thinking verbs
    "think": 4060, "feel": 4061, "resonate": 4062, "dream": 4063,
    "crystallize": 4064, "dissolve": 4065, "bind": 4066, "bundle": 4067,
}


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5000-5999: RELATIONSHIPS
# ═══════════════════════════════════════════════════════════════════════════════

RELATION_DIMENSIONS = {
    # Inheritance
    "extends": 5000, "implements": 5001, "inherits": 5002,

    # Composition
    "contains": 5010, "has": 5011, "owns": 5012,

    # Dependencies
    "imports": 5020, "uses": 5021, "depends_on": 5022,

    # Calls
    "calls": 5030, "invokes": 5031, "triggers": 5032,

    # Data flow
    "produces": 5040, "consumes": 5041, "transforms": 5042,

    # Graph
    "connects": 5050, "links": 5051, "references": 5052,
}


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 6000-6999: QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

QUALITY_DIMENSIONS = {
    # Size
    "small": 6000, "medium": 6001, "large": 6002,

    # Complexity
    "simple": 6010, "moderate": 6011, "complex": 6012,

    # Style
    "documented": 6020, "typed": 6021, "tested": 6022,

    # State
    "stable": 6030, "experimental": 6031, "deprecated": 6032,
}


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

CODE_SCHEMA_10K: Dict[str, int] = {}

# Populate from all blocks
for prefix, block in [
    ("import:", {**STDLIB_IMPORTS, **THIRDPARTY_IMPORTS, **ADA_IMPORTS}),
    ("pattern:", CODE_PATTERNS),
    ("type:", TYPE_DIMENSIONS),
    ("domain:", DOMAIN_DIMENSIONS),
    ("action:", ACTION_DIMENSIONS),
    ("relation:", RELATION_DIMENSIONS),
    ("quality:", QUALITY_DIMENSIONS),
]:
    for name, dim in block.items():
        CODE_SCHEMA_10K[f"{prefix}{name}"] = dim


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class CodeSchemaEncoder:
    """
    Encodes CodeChunks to 10kD vectors using the code schema.

    No LLM required. Structure maps directly to dimensions.
    """

    def __init__(self, schema: Optional[Dict[str, int]] = None):
        self.schema = schema or CODE_SCHEMA_10K
        self.dynamic_next = DYNAMIC_START
        self.learned: Dict[str, int] = {}

    def encode(self, chunk: CodeChunk) -> np.ndarray:
        """
        Encode a code chunk to 10kD vector.

        Uses structural features only (no LLM).
        """
        vec = np.zeros(10000, dtype=np.float32)

        # 1. Encode imports
        for imp in chunk.imports:
            # Try full import
            dim = self._get_import_dim(imp)
            if dim is not None:
                vec[dim] = 1.0

        # 2. Encode patterns
        self._encode_patterns(chunk, vec)

        # 3. Encode types (from args, returns)
        self._encode_types(chunk, vec)

        # 4. Encode domains (from name, docstring)
        self._encode_domains(chunk, vec)

        # 5. Encode actions (from function name)
        self._encode_actions(chunk, vec)

        # 6. Encode relationships
        self._encode_relations(chunk, vec)

        # 7. Encode quality
        self._encode_quality(chunk, vec)

        return vec

    def _get_import_dim(self, imp: str) -> Optional[int]:
        """Get dimension for an import."""
        # Check full import
        key = f"import:{imp}"
        if key in self.schema:
            return self.schema[key]

        # Check just the module part
        parts = imp.split('.')
        for i in range(len(parts)):
            partial = '.'.join(parts[:i+1])
            key = f"import:{partial}"
            if key in self.schema:
                return self.schema[key]

            # Also check without prefix
            if partial in STDLIB_IMPORTS:
                return STDLIB_IMPORTS[partial]
            if partial in THIRDPARTY_IMPORTS:
                return THIRDPARTY_IMPORTS[partial]
            if partial in ADA_IMPORTS:
                return ADA_IMPORTS[partial]

        return None

    def _encode_patterns(self, chunk: CodeChunk, vec: np.ndarray):
        """Encode structural patterns."""
        # Async
        if chunk.name.startswith("async_") or "async" in chunk.decorators:
            vec[CODE_PATTERNS["async"]] = 1.0

        # Dataclass
        if "dataclass" in chunk.decorators:
            vec[CODE_PATTERNS["dataclass"]] = 1.0

        # DTO
        if "DTO" in chunk.name or "Dto" in chunk.name:
            vec[CODE_PATTERNS["dto"]] = 1.0

        # Test
        if chunk.name.startswith("test_"):
            vec[CODE_PATTERNS["test_"]] = 1.0

        # Decorator
        if chunk.decorators:
            vec[CODE_PATTERNS["decorator"]] = 1.0

        # Generator (has yield)
        if "yield" in str(chunk.code_preview):
            vec[CODE_PATTERNS["generator"]] = 1.0

    def _encode_types(self, chunk: CodeChunk, vec: np.ndarray):
        """Encode type information."""
        # From return type
        if chunk.returns:
            ret_lower = chunk.returns.lower()
            for type_name, dim in TYPE_DIMENSIONS.items():
                if type_name in ret_lower:
                    vec[dim] = 1.0

        # From args
        for arg in chunk.args:
            arg_lower = arg.lower()
            for type_name, dim in TYPE_DIMENSIONS.items():
                if type_name in arg_lower:
                    vec[dim] = 0.5

    def _encode_domains(self, chunk: CodeChunk, vec: np.ndarray):
        """Encode domain from name and docstring."""
        text = f"{chunk.name} {chunk.docstring or ''}".lower()

        for domain, dim in DOMAIN_DIMENSIONS.items():
            if domain in text:
                vec[dim] = 1.0

    def _encode_actions(self, chunk: CodeChunk, vec: np.ndarray):
        """Encode actions from function name."""
        name_lower = chunk.name.lower()

        for action, dim in ACTION_DIMENSIONS.items():
            if name_lower.startswith(action) or f"_{action}" in name_lower:
                vec[dim] = 1.0

    def _encode_relations(self, chunk: CodeChunk, vec: np.ndarray):
        """Encode relationships."""
        # Inheritance
        if chunk.inherits:
            vec[RELATION_DIMENSIONS["inherits"]] = 1.0
            vec[RELATION_DIMENSIONS["extends"]] = len(chunk.inherits) / 5.0

        # Contains
        if chunk.contains:
            vec[RELATION_DIMENSIONS["contains"]] = len(chunk.contains) / 20.0

        # Calls
        if chunk.calls:
            vec[RELATION_DIMENSIONS["calls"]] = min(1.0, len(chunk.calls) / 10.0)

        # Imports
        if chunk.imports:
            vec[RELATION_DIMENSIONS["imports"]] = min(1.0, len(chunk.imports) / 20.0)

    def _encode_quality(self, chunk: CodeChunk, vec: np.ndarray):
        """Encode quality indicators."""
        # Size
        lines = chunk.line_end - chunk.line_start
        if lines < 20:
            vec[QUALITY_DIMENSIONS["small"]] = 1.0
        elif lines < 100:
            vec[QUALITY_DIMENSIONS["medium"]] = 1.0
        else:
            vec[QUALITY_DIMENSIONS["large"]] = 1.0

        # Documented
        if chunk.docstring:
            vec[QUALITY_DIMENSIONS["documented"]] = 1.0

        # Typed (has return annotation or type hints)
        if chunk.returns:
            vec[QUALITY_DIMENSIONS["typed"]] = 1.0

    def learn_concept(self, name: str) -> int:
        """
        Learn a new concept and assign it a dynamic dimension.

        Returns the assigned dimension.
        """
        if name in self.learned:
            return self.learned[name]

        if self.dynamic_next >= DYNAMIC_END:
            # Reuse oldest
            self.dynamic_next = DYNAMIC_START

        dim = self.dynamic_next
        self.learned[name] = dim
        self.schema[f"learned:{name}"] = dim
        self.dynamic_next += 1

        return dim

    def encode_batch(self, chunks: List[CodeChunk]) -> np.ndarray:
        """Encode multiple chunks to matrix."""
        return np.array([self.encode(c) for c in chunks])

    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        return {
            "schema_size": len(self.schema),
            "learned_concepts": len(self.learned),
            "dynamic_next": self.dynamic_next,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_schema():
    """Test the code schema encoder."""
    print("=" * 60)
    print("CODE SCHEMA 10K TEST")
    print("=" * 60)

    # Create a sample chunk
    chunk = CodeChunk(
        id="test.py:function:compute_resonance",
        chunk_type=ChunkType.FUNCTION,
        name="compute_resonance",
        file_path="test.py",
        line_start=10,
        line_end=50,
        docstring="Compute VSA resonance between vectors.",
        imports=["numpy", "typing", "extension.agi_thinking.vsa_utils"],
        calls=["np.dot", "normalize", "bundle"],
        args=["vec_a", "vec_b", "threshold"],
        returns="float",
        decorators=[],
    )

    encoder = CodeSchemaEncoder()
    vec = encoder.encode(chunk)

    print(f"\nEncoded chunk: {chunk.name}")
    print(f"  Non-zero dimensions: {np.count_nonzero(vec)}")

    # Show which dimensions are active
    active = np.where(vec > 0)[0]
    print(f"\n  Active dimensions:")

    # Reverse lookup
    dim_to_name = {v: k for k, v in CODE_SCHEMA_10K.items()}

    for dim in active[:20]:
        name = dim_to_name.get(dim, f"dim_{dim}")
        print(f"    [{dim:5d}] {name}: {vec[dim]:.2f}")

    print(f"\n  Stats: {encoder.get_stats()}")

    print("\n" + "=" * 60)
    print("SCHEMA TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_schema()
