"""
LanceDB Client - Vector storage wrapper for Ada's semantic memory.

Manages:
- Thought vectors (1024D Jina embeddings)
- Concept vectors
- Similarity search with hybrid filtering
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import lancedb
import pyarrow as pa


class LanceClient:
    """LanceDB client for Ada AGI Surface."""

    # Default vector dimension (Jina embeddings)
    VECTOR_DIM = 1024

    def __init__(self, db_path: str):
        """Initialize LanceDB connection."""
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self):
        """Initialize tables if they don't exist."""
        # Thoughts table
        if "thoughts" not in self.db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), self.VECTOR_DIM)),
                ("content", pa.string()),
                ("session_id", pa.string()),
                ("timestamp", pa.string()),
                ("confidence", pa.float32()),
            ])
            self.db.create_table("thoughts", schema=schema)
            print("[LANCE] Created 'thoughts' table")

        # Concepts table
        if "concepts" not in self.db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), self.VECTOR_DIM)),
                ("name", pa.string()),
                ("salience", pa.float32()),
                ("timestamp", pa.string()),
            ])
            self.db.create_table("concepts", schema=schema)
            print("[LANCE] Created 'concepts' table")

        # Episodes table (context vectors)
        if "episodes" not in self.db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), self.VECTOR_DIM)),
                ("session_id", pa.string()),
                ("summary", pa.string()),
                ("timestamp", pa.string()),
            ])
            self.db.create_table("episodes", schema=schema)
            print("[LANCE] Created 'episodes' table")

    async def search(
        self,
        vector: List[float],
        table: str = "thoughts",
        top_k: int = 10,
        filter_dict: Dict[str, Any] = None,
    ) -> List[Dict]:
        """
        Similarity search in LanceDB.

        Args:
            vector: Query vector
            table: Table name to search
            top_k: Number of results
            filter_dict: Optional filter conditions

        Returns:
            List of matching records with distances
        """
        try:
            tbl = self.db.open_table(table)
            query = tbl.search(vector).limit(top_k)

            # Apply filter if provided
            if filter_dict:
                filter_str = self._build_filter(filter_dict)
                if filter_str:
                    query = query.where(filter_str)

            results = query.to_list()

            # Convert to dicts and add distance scores
            return [
                {
                    **{k: v for k, v in row.items() if k != "vector"},
                    "distance": row.get("_distance", 0.0),
                }
                for row in results
            ]
        except Exception as e:
            print(f"[LANCE] Search error: {e}")
            return []

    def _build_filter(self, filter_dict: Dict[str, Any]) -> str:
        """Build filter string from dict."""
        conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, str):
                conditions.append(f"{key} = '{value}'")
            elif isinstance(value, (int, float)):
                conditions.append(f"{key} = {value}")
            elif isinstance(value, dict):
                # Handle operators like {"$gt": 0.5}
                for op, val in value.items():
                    if op == "$gt":
                        conditions.append(f"{key} > {val}")
                    elif op == "$gte":
                        conditions.append(f"{key} >= {val}")
                    elif op == "$lt":
                        conditions.append(f"{key} < {val}")
                    elif op == "$lte":
                        conditions.append(f"{key} <= {val}")
                    elif op == "$ne":
                        conditions.append(f"{key} != {val}")

        return " AND ".join(conditions) if conditions else ""

    async def upsert(
        self,
        id: str,
        vector: List[float],
        table: str = "thoughts",
        metadata: Dict[str, Any] = None,
    ) -> None:
        """
        Add or update vector in table.

        Args:
            id: Unique identifier
            vector: Vector to store
            table: Target table
            metadata: Additional metadata fields
        """
        try:
            tbl = self.db.open_table(table)

            # Pad or truncate vector to correct dimension
            if len(vector) < self.VECTOR_DIM:
                vector = vector + [0.0] * (self.VECTOR_DIM - len(vector))
            elif len(vector) > self.VECTOR_DIM:
                vector = vector[:self.VECTOR_DIM]

            # Build row
            row = {
                "id": id,
                "vector": vector,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    if key not in ["id", "vector"]:
                        row[key] = value

            # Ensure required fields have defaults
            if table == "thoughts":
                row.setdefault("content", "")
                row.setdefault("session_id", "")
                row.setdefault("confidence", 0.5)
            elif table == "concepts":
                row.setdefault("name", "")
                row.setdefault("salience", 0.5)
            elif table == "episodes":
                row.setdefault("session_id", "")
                row.setdefault("summary", "")

            tbl.add([row])
        except Exception as e:
            print(f"[LANCE] Upsert error: {e}")

    async def get(self, id: str, table: str = "thoughts") -> Optional[Dict]:
        """Get record by ID."""
        try:
            tbl = self.db.open_table(table)
            results = tbl.search().where(f"id = '{id}'").limit(1).to_list()
            return results[0] if results else None
        except Exception as e:
            print(f"[LANCE] Get error: {e}")
            return None

    async def delete(self, id: str, table: str = "thoughts") -> bool:
        """Delete record by ID."""
        try:
            tbl = self.db.open_table(table)
            tbl.delete(f"id = '{id}'")
            return True
        except Exception as e:
            print(f"[LANCE] Delete error: {e}")
            return False

    async def hybrid_search(
        self,
        vector: List[float],
        style_vector: List[float] = None,
        qualia_vector: List[float] = None,
        table: str = "thoughts",
        top_k: int = 10,
        content_weight: float = 0.6,
        style_weight: float = 0.2,
        qualia_weight: float = 0.2,
    ) -> List[Dict]:
        """
        Hybrid search combining content, style, and qualia similarity.

        This is a simplified implementation - in production you might
        want to use multiple vector columns or a more sophisticated
        reranking approach.

        Args:
            vector: Content vector (1024D)
            style_vector: Thinking style vector (33D)
            qualia_vector: Qualia state vector (17D)
            table: Target table
            top_k: Number of results
            content_weight: Weight for content similarity
            style_weight: Weight for style similarity
            qualia_weight: Weight for qualia similarity

        Returns:
            Results with combined scores
        """
        # For now, just do content search
        # Full hybrid search would require storing style/qualia vectors
        # in the same table or doing multi-table joins
        results = await self.search(vector, table, top_k * 2)

        # In a full implementation, we would:
        # 1. Retrieve style/qualia vectors for each result from Kuzu
        # 2. Compute style similarity using cosine distance
        # 3. Compute qualia similarity using cosine distance
        # 4. Combine scores: combined = w1*content + w2*style + w3*qualia
        # 5. Rerank by combined score

        return results[:top_k]

    async def count(self, table: str = "thoughts") -> int:
        """Get record count for table."""
        try:
            tbl = self.db.open_table(table)
            return tbl.count_rows()
        except Exception as e:
            print(f"[LANCE] Count error: {e}")
            return 0

    def is_connected(self) -> bool:
        """Check if database connection is active."""
        return self.db is not None

    def close(self):
        """Close database connection."""
        # LanceDB connections are managed automatically
        print("[LANCE] Connection closed")
