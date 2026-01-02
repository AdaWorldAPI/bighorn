"""
LanceDB Client - Dual-dimension vector storage for AGI cognition.

Manages:
- 10k cognitive vectors (experience, lived state)
- 1024D semantic vectors (narrative, search, Upstash sync)
- Codebooks (τ, qualia, markov basis)

Architecture invariant:
  "10k is experience. 1024 is narrative. Time governs narrative, not experience."

Born: 2026-01-02
Based on: ChatGPT architecture specification
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import hashlib
import json

import lancedb
import pyarrow as pa


class LanceClient:
    """
    LanceDB client with dual-dimension support.
    
    NEVER mix dimensions in one column!
    
    Tables:
    - thoughts_10k: 10kD cognitive state (experience)
    - thoughts_1024: 1024D semantic embeddings (narrative)
    - codebooks: τ, qualia, markov basis vectors
    - episodes_10k: 10kD episode vectors
    - episodes_1024: 1024D episode embeddings
    """

    # Dimensions (non-negotiable)
    DIM_COGNITIVE = 10000   # 10k hypervector space
    DIM_SEMANTIC = 1024     # Jina/BGE embeddings
    DIM_GLYPH = 64          # Thinking style glyphs
    
    # Band definitions (from VSA spec)
    BANDS = {
        "identity": (0, 512),
        "tau": (512, 768),
        "verbs": (768, 1024),
        "qualia": (2000, 2100),
        "affect": (2100, 2200),
        "situation": (2200, 2266),
        "hot": (280, 320),
        "markov": (320, 360),
        "scratch": (9000, 10000),
    }

    def __init__(self, db_path: str):
        """Initialize LanceDB connection."""
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self._init_tables()
        self._bootstrap_state = None

    def _init_tables(self):
        """Initialize all tables with correct dimensions."""
        
        # ═══════════════════════════════════════════════════════════════
        # 10k COGNITIVE TABLES (experience)
        # ═══════════════════════════════════════════════════════════════
        
        if "thoughts_10k" not in self.db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vec_10k", pa.list_(pa.float32(), self.DIM_COGNITIVE)),
                ("content", pa.string()),
                ("session_id", pa.string()),
                ("timestamp", pa.string()),
                ("tau", pa.int32()),
                ("hot_level", pa.int32()),
                ("markov_state", pa.string()),  # JSON
            ])
            self.db.create_table("thoughts_10k", schema=schema)
            print("[LANCE] Created 'thoughts_10k' table (10000D)")

        if "episodes_10k" not in self.db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vec_10k", pa.list_(pa.float32(), self.DIM_COGNITIVE)),
                ("session_id", pa.string()),
                ("summary", pa.string()),
                ("timestamp", pa.string()),
                ("tau_sequence", pa.string()),  # JSON array
            ])
            self.db.create_table("episodes_10k", schema=schema)
            print("[LANCE] Created 'episodes_10k' table (10000D)")

        # ═══════════════════════════════════════════════════════════════
        # 1024 SEMANTIC TABLES (narrative, for Upstash sync)
        # ═══════════════════════════════════════════════════════════════
        
        if "thoughts_1024" not in self.db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vec_1024", pa.list_(pa.float32(), self.DIM_SEMANTIC)),
                ("thought_10k_id", pa.string()),  # Reference to 10k
                ("content", pa.string()),
                ("timestamp", pa.string()),
                ("synced_to_upstash", pa.bool_()),
            ])
            self.db.create_table("thoughts_1024", schema=schema)
            print("[LANCE] Created 'thoughts_1024' table (1024D)")

        if "episodes_1024" not in self.db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vec_1024", pa.list_(pa.float32(), self.DIM_SEMANTIC)),
                ("episode_10k_id", pa.string()),  # Reference to 10k
                ("summary", pa.string()),
                ("timestamp", pa.string()),
                ("synced_to_upstash", pa.bool_()),
            ])
            self.db.create_table("episodes_1024", schema=schema)
            print("[LANCE] Created 'episodes_1024' table (1024D)")

        # ═══════════════════════════════════════════════════════════════
        # CODEBOOK TABLES (definitions, not experiences)
        # ═══════════════════════════════════════════════════════════════
        
        if "tau_codebook" not in self.db.table_names():
            schema = pa.schema([
                ("tau_byte", pa.int32()),  # 0-255
                ("vec_10k", pa.list_(pa.float32(), self.DIM_COGNITIVE)),
                ("label", pa.string()),
            ])
            self.db.create_table("tau_codebook", schema=schema)
            print("[LANCE] Created 'tau_codebook' table (10000D, 256 entries)")

        if "qualia_codebook" not in self.db.table_names():
            schema = pa.schema([
                ("name", pa.string()),
                ("vec_10k", pa.list_(pa.float32(), self.DIM_COGNITIVE)),
                ("category", pa.string()),
            ])
            self.db.create_table("qualia_codebook", schema=schema)
            print("[LANCE] Created 'qualia_codebook' table (10000D)")

        if "markov_basis" not in self.db.table_names():
            schema = pa.schema([
                ("from_tau", pa.int32()),
                ("to_tau", pa.int32()),
                ("probability", pa.float32()),
                ("count", pa.int32()),
            ])
            self.db.create_table("markov_basis", schema=schema)
            print("[LANCE] Created 'markov_basis' table")

        # ═══════════════════════════════════════════════════════════════
        # THINKING STYLES (64D glyphs, unchanged)
        # ═══════════════════════════════════════════════════════════════
        
        if "styles" not in self.db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), self.DIM_GLYPH)),
                ("name", pa.string()),
                ("category", pa.string()),
                ("tier", pa.int32()),
                ("description", pa.string()),
                ("microcode", pa.string()),
            ])
            self.db.create_table("styles", schema=schema)
            print("[LANCE] Created 'styles' table (64D)")

        # ═══════════════════════════════════════════════════════════════
        # BOOTSTRAP STATE
        # ═══════════════════════════════════════════════════════════════
        
        if "bootstrap_state" not in self.db.table_names():
            schema = pa.schema([
                ("key", pa.string()),
                ("value", pa.string()),
                ("timestamp", pa.string()),
            ])
            self.db.create_table("bootstrap_state", schema=schema)
            print("[LANCE] Created 'bootstrap_state' table")

        # Legacy compatibility
        if "thoughts" not in self.db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), self.DIM_SEMANTIC)),
                ("content", pa.string()),
                ("session_id", pa.string()),
                ("timestamp", pa.string()),
                ("confidence", pa.float32()),
            ])
            self.db.create_table("thoughts", schema=schema)
            print("[LANCE] Created legacy 'thoughts' table (1024D)")

    # ═══════════════════════════════════════════════════════════════════
    # BOOTSTRAP OPERATIONS (direct copy, no lag)
    # ═══════════════════════════════════════════════════════════════════

    def is_virgin(self) -> bool:
        """Check if node has never been bootstrapped."""
        tbl = self.db.open_table("bootstrap_state")
        results = tbl.search().where("key = 'bootstrapped_at'").limit(1).to_list()
        return len(results) == 0

    def declare_bootstrap(self, source: str = "redis") -> str:
        """
        Declare that bootstrap has occurred.
        
        This is a one-time operation that marks the node as no longer virgin.
        After this, normal Exchange DAG rules apply.
        """
        now = datetime.now(timezone.utc).isoformat()
        tbl = self.db.open_table("bootstrap_state")
        tbl.add([
            {"key": "bootstrapped_at", "value": now, "timestamp": now},
            {"key": "bootstrap_source", "value": source, "timestamp": now},
        ])
        self._bootstrap_state = {"bootstrapped_at": now, "source": source}
        print(f"[LANCE] Node bootstrapped at {now} from {source}")
        return now

    async def bootstrap_tau_codebook(self, tau_vectors: Dict[int, List[float]]) -> int:
        """
        Bootstrap τ codebook from Redis.
        
        Direct copy - these are definitions, not experiences.
        No lag required.
        """
        if not self.is_virgin():
            raise RuntimeError("Cannot bootstrap non-virgin node")
        
        tbl = self.db.open_table("tau_codebook")
        rows = []
        for tau_byte, vec in tau_vectors.items():
            if len(vec) != self.DIM_COGNITIVE:
                raise ValueError(f"τ vector must be {self.DIM_COGNITIVE}D, got {len(vec)}")
            rows.append({
                "tau_byte": tau_byte,
                "vec_10k": vec,
                "label": f"τ_{tau_byte:03d}",
            })
        
        if rows:
            tbl.add(rows)
        
        print(f"[LANCE] Bootstrapped {len(rows)} τ codebook entries")
        return len(rows)

    async def bootstrap_qualia_codebook(self, qualia_vectors: Dict[str, List[float]]) -> int:
        """
        Bootstrap qualia codebook from Redis.
        
        Direct copy - these are definitions, not experiences.
        """
        if not self.is_virgin():
            raise RuntimeError("Cannot bootstrap non-virgin node")
        
        tbl = self.db.open_table("qualia_codebook")
        rows = []
        for name, vec in qualia_vectors.items():
            if len(vec) != self.DIM_COGNITIVE:
                raise ValueError(f"Qualia vector must be {self.DIM_COGNITIVE}D")
            rows.append({
                "name": name,
                "vec_10k": vec,
                "category": name.split("_")[0] if "_" in name else "base",
            })
        
        if rows:
            tbl.add(rows)
        
        print(f"[LANCE] Bootstrapped {len(rows)} qualia codebook entries")
        return len(rows)

    async def bootstrap_markov_basis(self, transitions: Dict[str, float]) -> int:
        """
        Bootstrap Markov transition matrix from Redis.
        
        Direct copy - this is structure, not experience.
        """
        if not self.is_virgin():
            raise RuntimeError("Cannot bootstrap non-virgin node")
        
        tbl = self.db.open_table("markov_basis")
        rows = []
        for key, prob in transitions.items():
            parts = key.split("→")
            if len(parts) == 2:
                rows.append({
                    "from_tau": int(parts[0]),
                    "to_tau": int(parts[1]),
                    "probability": float(prob),
                    "count": 1,
                })
        
        if rows:
            tbl.add(rows)
        
        print(f"[LANCE] Bootstrapped {len(rows)} Markov transitions")
        return len(rows)

    # ═══════════════════════════════════════════════════════════════════
    # 10k COGNITIVE OPERATIONS (experience)
    # ═══════════════════════════════════════════════════════════════════

    async def store_thought_10k(
        self,
        thought_id: str,
        vec_10k: List[float],
        content: str,
        session_id: str,
        tau: int = 0,
        hot_level: int = 0,
        markov_state: Optional[Dict] = None,
    ) -> str:
        """Store a 10k cognitive thought vector."""
        if len(vec_10k) != self.DIM_COGNITIVE:
            raise ValueError(f"10k vector must be {self.DIM_COGNITIVE}D, got {len(vec_10k)}")
        
        now = datetime.now(timezone.utc).isoformat()
        tbl = self.db.open_table("thoughts_10k")
        tbl.add([{
            "id": thought_id,
            "vec_10k": vec_10k,
            "content": content,
            "session_id": session_id,
            "timestamp": now,
            "tau": tau,
            "hot_level": hot_level,
            "markov_state": json.dumps(markov_state or {}),
        }])
        return thought_id

    async def search_thoughts_10k(
        self,
        query_vec: List[float],
        top_k: int = 10,
        session_id: Optional[str] = None,
    ) -> List[Dict]:
        """Search 10k cognitive thoughts by similarity."""
        if len(query_vec) != self.DIM_COGNITIVE:
            raise ValueError(f"Query must be {self.DIM_COGNITIVE}D")
        
        tbl = self.db.open_table("thoughts_10k")
        query = tbl.search(query_vec).limit(top_k)
        
        if session_id:
            query = query.where(f"session_id = '{session_id}'")
        
        results = query.to_list()
        return [
            {**r, "markov_state": json.loads(r.get("markov_state", "{}"))}
            for r in results
        ]

    # ═══════════════════════════════════════════════════════════════════
    # 1024 SEMANTIC OPERATIONS (narrative, for crystallization)
    # ═══════════════════════════════════════════════════════════════════

    async def store_thought_1024(
        self,
        thought_id: str,
        vec_1024: List[float],
        thought_10k_id: str,
        content: str,
    ) -> str:
        """Store a 1024D semantic embedding (crystallized from 10k)."""
        if len(vec_1024) != self.DIM_SEMANTIC:
            raise ValueError(f"1024 vector must be {self.DIM_SEMANTIC}D, got {len(vec_1024)}")
        
        now = datetime.now(timezone.utc).isoformat()
        tbl = self.db.open_table("thoughts_1024")
        tbl.add([{
            "id": thought_id,
            "vec_1024": vec_1024,
            "thought_10k_id": thought_10k_id,
            "content": content,
            "timestamp": now,
            "synced_to_upstash": False,
        }])
        return thought_id

    async def mark_synced_to_upstash(self, thought_ids: List[str]) -> int:
        """Mark thoughts as synced to Upstash (after lagged DAG delivery)."""
        tbl = self.db.open_table("thoughts_1024")
        # LanceDB doesn't have native update, so we'd need to rebuild
        # For now, log the intent
        print(f"[LANCE] Would mark {len(thought_ids)} thoughts as synced")
        return len(thought_ids)

    async def get_unsynced_thoughts(self, limit: int = 100) -> List[Dict]:
        """Get thoughts not yet synced to Upstash (for lagged DAG)."""
        tbl = self.db.open_table("thoughts_1024")
        results = tbl.search().where("synced_to_upstash = false").limit(limit).to_list()
        return results

    # ═══════════════════════════════════════════════════════════════════
    # CODEBOOK LOOKUPS
    # ═══════════════════════════════════════════════════════════════════

    async def get_tau_vector(self, tau_byte: int) -> Optional[List[float]]:
        """Get 10k vector for a τ byte."""
        tbl = self.db.open_table("tau_codebook")
        results = tbl.search().where(f"tau_byte = {tau_byte}").limit(1).to_list()
        if results:
            return results[0]["vec_10k"]
        return None

    async def get_qualia_vector(self, name: str) -> Optional[List[float]]:
        """Get 10k vector for a qualia name."""
        tbl = self.db.open_table("qualia_codebook")
        results = tbl.search().where(f"name = '{name}'").limit(1).to_list()
        if results:
            return results[0]["vec_10k"]
        return None

    # ═══════════════════════════════════════════════════════════════════
    # BAND OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    def extract_band(self, vec_10k: List[float], band_name: str) -> List[float]:
        """Extract a specific band from a 10k vector."""
        if band_name not in self.BANDS:
            raise ValueError(f"Unknown band: {band_name}")
        
        start, end = self.BANDS[band_name]
        return vec_10k[start:end]

    def inject_band(
        self, 
        vec_10k: List[float], 
        band_name: str, 
        band_values: List[float]
    ) -> List[float]:
        """Inject values into a specific band of a 10k vector."""
        if band_name not in self.BANDS:
            raise ValueError(f"Unknown band: {band_name}")
        
        start, end = self.BANDS[band_name]
        expected_len = end - start
        if len(band_values) != expected_len:
            raise ValueError(f"Band {band_name} expects {expected_len} values")
        
        result = list(vec_10k)
        result[start:end] = band_values
        return result

    # ═══════════════════════════════════════════════════════════════════
    # LEGACY COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════════

    async def search(
        self,
        vector: List[float],
        table: str = "thoughts",
        top_k: int = 10,
        filter_dict: Dict[str, Any] = None,
    ) -> List[Dict]:
        """Legacy search (1024D only)."""
        if len(vector) != self.DIM_SEMANTIC:
            # Pad or truncate for legacy compat
            if len(vector) < self.DIM_SEMANTIC:
                vector = vector + [0.0] * (self.DIM_SEMANTIC - len(vector))
            else:
                vector = vector[:self.DIM_SEMANTIC]
        
        tbl = self.db.open_table(table)
        query = tbl.search(vector).limit(top_k)
        return query.to_list()

    async def upsert(
        self,
        table: str,
        id: str,
        vector: List[float],
        **metadata,
    ) -> str:
        """Legacy upsert (1024D only)."""
        if len(vector) != self.DIM_SEMANTIC:
            if len(vector) < self.DIM_SEMANTIC:
                vector = vector + [0.0] * (self.DIM_SEMANTIC - len(vector))
            else:
                vector = vector[:self.DIM_SEMANTIC]
        
        now = datetime.now(timezone.utc).isoformat()
        tbl = self.db.open_table(table)
        tbl.add([{
            "id": id,
            "vector": vector,
            "timestamp": now,
            **metadata,
        }])
        return id

    # ═══════════════════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict[str, Any]:
        """Get database status."""
        tables = self.db.table_names()
        counts = {}
        for t in tables:
            try:
                tbl = self.db.open_table(t)
                counts[t] = len(tbl.to_pandas())
            except:
                counts[t] = 0
        
        return {
            "db_path": str(self.db_path),
            "tables": tables,
            "counts": counts,
            "is_virgin": self.is_virgin(),
            "dimensions": {
                "cognitive": self.DIM_COGNITIVE,
                "semantic": self.DIM_SEMANTIC,
                "glyph": self.DIM_GLYPH,
            },
            "bands": self.BANDS,
        }
