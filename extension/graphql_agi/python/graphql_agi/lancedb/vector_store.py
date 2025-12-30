"""
LanceDB Vector Store Integration

Provides vector storage and similarity search using LanceDB.
Supports multiple embedding providers and hybrid search.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for the vector store."""
    uri: str = "lance://memory"
    dimension: int = 1536
    distance_metric: str = "cosine"  # cosine, l2, dot
    embedding_model: str = "text-embedding-3-large"
    embedding_api_key: Optional[str] = None
    embedding_provider: str = "openai"  # openai, voyage, cohere, ollama

    # Index settings
    index_type: str = "IVF_PQ"  # IVF_PQ, IVF_FLAT, HNSW
    num_partitions: int = 256
    num_sub_vectors: int = 96

    # HNSW settings
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200

    # Connection settings
    max_connections: int = 10
    batch_size: int = 1000


@dataclass
class VectorSearchResult:
    """Result from a vector search."""
    entity_id: str
    entity_type: str
    score: float
    content: Optional[str] = None
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingProvider:
    """Base class for embedding providers."""

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    def embed(self, text: str) -> List[float]:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except ImportError:
            logger.warning("OpenAI package not installed. Using mock embeddings.")
            return self._mock_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [d.embedding for d in response.data]
        except ImportError:
            return [self._mock_embedding(t) for t in texts]

    def _mock_embedding(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding for testing."""
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Create a 1536-dimensional vector from the hash
        embedding = []
        for i in range(1536):
            byte_idx = i % 32
            embedding.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)
        return embedding


class OllamaEmbedding(EmbeddingProvider):
    """Ollama local embedding provider."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> List[float]:
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            return response.json()["embedding"]
        except Exception as e:
            logger.warning(f"Ollama embedding failed: {e}. Using mock.")
            return self._mock_embedding(text)

    def _mock_embedding(self, text: str) -> List[float]:
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(1536):
            byte_idx = i % 32
            embedding.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)
        return embedding


class VectorStore:
    """
    LanceDB-based Vector Store

    Provides vector storage, indexing, and similarity search functionality.
    Integrates with multiple embedding providers.

    Example:
        >>> config = VectorStoreConfig(
        ...     uri="lance://./vectors",
        ...     embedding_model="text-embedding-3-large"
        ... )
        >>> store = VectorStore(config)
        >>>
        >>> # Store a vector
        >>> store.store("user_1", "User", vector=embedding, content="John Doe")
        >>>
        >>> # Search for similar
        >>> results = store.search(query_vector, "User", top_k=5)
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._db = None
        self._tables: Dict[str, Any] = {}
        self._embedding_provider: Optional[EmbeddingProvider] = None

        self._connect()
        self._setup_embedding_provider()

    def _connect(self) -> None:
        """Connect to LanceDB."""
        try:
            import lancedb
            if self.config.uri.startswith("lance://memory"):
                self._db = lancedb.connect(":memory:")
            elif self.config.uri.startswith("lance://"):
                path = self.config.uri[8:]
                self._db = lancedb.connect(path)
            else:
                self._db = lancedb.connect(self.config.uri)
            logger.info(f"Connected to LanceDB at {self.config.uri}")
        except ImportError:
            logger.warning("LanceDB not installed. Using in-memory mock store.")
            self._db = None

    def _setup_embedding_provider(self) -> None:
        """Setup the embedding provider based on config."""
        provider = self.config.embedding_provider.lower()

        if provider == "openai":
            self._embedding_provider = OpenAIEmbedding(
                model=self.config.embedding_model,
                api_key=self.config.embedding_api_key
            )
        elif provider == "ollama":
            self._embedding_provider = OllamaEmbedding(
                model=self.config.embedding_model
            )
        else:
            # Default to OpenAI
            self._embedding_provider = OpenAIEmbedding(
                model=self.config.embedding_model,
                api_key=self.config.embedding_api_key
            )

    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self._embedding_provider:
            return self._embedding_provider.embed(text)
        # Fallback mock embedding
        return OpenAIEmbedding("mock", None)._mock_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self._embedding_provider:
            return self._embedding_provider.embed_batch(texts)
        return [self.embed(t) for t in texts]

    def _get_table(self, entity_type: str):
        """Get or create a table for an entity type."""
        table_name = f"vectors_{entity_type.lower()}"

        if table_name in self._tables:
            return self._tables[table_name]

        if self._db is None:
            # Mock table
            self._tables[table_name] = []
            return self._tables[table_name]

        try:
            # Check if table exists
            if table_name in self._db.table_names():
                self._tables[table_name] = self._db.open_table(table_name)
            else:
                # Create table with schema
                import pyarrow as pa
                schema = pa.schema([
                    pa.field("entity_id", pa.string()),
                    pa.field("entity_type", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.config.dimension)),
                    pa.field("content", pa.string()),
                    pa.field("metadata", pa.string()),
                ])
                self._tables[table_name] = self._db.create_table(
                    table_name,
                    schema=schema
                )
            return self._tables[table_name]
        except Exception as e:
            logger.error(f"Failed to get/create table: {e}")
            self._tables[table_name] = []
            return self._tables[table_name]

    def store(
        self,
        entity_id: str,
        entity_type: str,
        vector: List[float],
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a vector for an entity.

        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity
            vector: The embedding vector
            content: Optional text content
            metadata: Optional metadata

        Returns:
            True if successful
        """
        table = self._get_table(entity_type)

        import json
        data = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "vector": vector,
            "content": content or "",
            "metadata": json.dumps(metadata or {})
        }

        if isinstance(table, list):
            # Mock store
            table.append(data)
            return True

        try:
            table.add([data])
            return True
        except Exception as e:
            logger.error(f"Failed to store vector: {e}")
            return False

    def store_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> int:
        """
        Store multiple vectors in batch.

        Args:
            items: List of dicts with entity_id, entity_type, vector, content, metadata

        Returns:
            Number of items stored
        """
        import json
        from collections import defaultdict

        # Group by entity type
        by_type = defaultdict(list)
        for item in items:
            by_type[item["entity_type"]].append({
                "entity_id": item["entity_id"],
                "entity_type": item["entity_type"],
                "vector": item["vector"],
                "content": item.get("content", ""),
                "metadata": json.dumps(item.get("metadata", {}))
            })

        count = 0
        for entity_type, type_items in by_type.items():
            table = self._get_table(entity_type)

            if isinstance(table, list):
                table.extend(type_items)
                count += len(type_items)
            else:
                try:
                    table.add(type_items)
                    count += len(type_items)
                except Exception as e:
                    logger.error(f"Batch store failed: {e}")

        return count

    def search(
        self,
        query_vector: List[float],
        table: str = "*",
        top_k: int = 10,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            table: Entity type to search (or "*" for all)
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Additional filters

        Returns:
            List of search results
        """
        results = []

        if table == "*":
            # Search all tables
            tables_to_search = list(self._tables.keys())
        else:
            tables_to_search = [f"vectors_{table.lower()}"]

        for table_name in tables_to_search:
            if table_name not in self._tables:
                continue

            tbl = self._tables[table_name]

            if isinstance(tbl, list):
                # Mock search
                for item in tbl:
                    score = self._cosine_similarity(query_vector, item["vector"])
                    if score >= threshold:
                        results.append(VectorSearchResult(
                            entity_id=item["entity_id"],
                            entity_type=item["entity_type"],
                            score=score,
                            content=item.get("content"),
                            metadata=json.loads(item.get("metadata", "{}"))
                        ))
            else:
                try:
                    search_results = tbl.search(query_vector).limit(top_k).to_list()
                    for r in search_results:
                        score = 1 - r.get("_distance", 0)  # Convert distance to similarity
                        if score >= threshold:
                            import json
                            results.append(VectorSearchResult(
                                entity_id=r["entity_id"],
                                entity_type=r["entity_type"],
                                score=score,
                                content=r.get("content"),
                                vector=r.get("vector") if filters and filters.get("include_vectors") else None,
                                metadata=json.loads(r.get("metadata", "{}"))
                            ))
                except Exception as e:
                    logger.error(f"Search failed: {e}")

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def hybrid_search(
        self,
        query: str,
        table: str,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        top_k: int = 10,
        text_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and text matching.

        Args:
            query: Search query text
            table: Table to search
            vector_weight: Weight for vector similarity
            text_weight: Weight for text matching
            top_k: Number of results
            text_fields: Fields for text search

        Returns:
            Combined search results
        """
        # Get vector search results
        query_vector = self.embed(query)
        vector_results = self.search(
            query_vector=query_vector,
            table=table,
            top_k=top_k * 2  # Get more for reranking
        )

        # Create result dict for combining scores
        combined = {}
        for r in vector_results:
            combined[r.entity_id] = {
                "entity_id": r.entity_id,
                "entity_type": r.entity_type,
                "content": r.content,
                "metadata": r.metadata,
                "vector_score": r.score,
                "text_score": 0.0
            }

        # Add text matching scores (simple keyword matching)
        query_terms = query.lower().split()
        for entity_id, item in combined.items():
            content = (item.get("content") or "").lower()
            matches = sum(1 for term in query_terms if term in content)
            item["text_score"] = matches / max(len(query_terms), 1)

        # Combine scores
        for item in combined.values():
            item["combined_score"] = (
                vector_weight * item["vector_score"] +
                text_weight * item["text_score"]
            )

        # Sort by combined score
        results = sorted(
            combined.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:top_k]

        return results

    def delete(self, entity_id: str, entity_type: str) -> bool:
        """Delete a vector by entity ID."""
        table = self._get_table(entity_type)

        if isinstance(table, list):
            initial_len = len(table)
            table[:] = [item for item in table if item["entity_id"] != entity_id]
            return len(table) < initial_len

        try:
            table.delete(f"entity_id = '{entity_id}'")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def create_index(self, entity_type: str) -> bool:
        """Create an index on a table."""
        table = self._get_table(entity_type)

        if isinstance(table, list):
            return True  # Mock - no index needed

        try:
            table.create_index(
                metric=self.config.distance_metric,
                num_partitions=self.config.num_partitions,
                num_sub_vectors=self.config.num_sub_vectors
            )
            return True
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            return False

    def count(self, entity_type: str) -> int:
        """Count vectors in a table."""
        table = self._get_table(entity_type)

        if isinstance(table, list):
            return len(table)

        try:
            return table.count_rows()
        except Exception:
            return 0

    def close(self) -> None:
        """Close the vector store connection."""
        self._tables.clear()
        self._db = None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
