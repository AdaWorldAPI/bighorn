"""
GraphQL AGI Client

The main entry point for GraphQL AGI functionality.
Provides a unified interface to GraphQL, LanceDB, AGI reasoning, and Ladybug debugging.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import kuzu

from ..lancedb.vector_store import VectorStore, VectorStoreConfig
from ..agi.reasoning import ReasoningEngine, ReasoningConfig, ReasoningStrategy
from ..agi.planning import PlanningSystem, PlanningConfig
from ..ladybug.debugger import LadybugDebugger, LadybugConfig

logger = logging.getLogger(__name__)


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"
    OPENAI_ADA = "text-embedding-ada-002"
    VOYAGE_2 = "voyage-2"
    VOYAGE_LARGE = "voyage-large-2"
    COHERE_ENGLISH = "embed-english-v3.0"
    COHERE_MULTILINGUAL = "embed-multilingual-v3.0"
    OLLAMA_NOMIC = "nomic-embed-text"
    OLLAMA_MXBAI = "mxbai-embed-large"


@dataclass
class GraphQLAGIConfig:
    """Configuration for GraphQL AGI client."""

    # Database settings
    database_path: str = ":memory:"
    read_only: bool = False
    buffer_pool_size: int = 0  # 0 = auto

    # GraphQL settings
    enable_introspection: bool = True
    max_query_depth: int = 15
    max_query_complexity: int = 1000
    enable_batching: bool = True

    # LanceDB / Vector settings
    lancedb_uri: str = "lance://memory"
    vector_dimension: int = 1536
    distance_metric: str = "cosine"  # cosine, l2, dot
    embedding_model: str = EmbeddingModel.OPENAI_LARGE

    # AGI settings
    enable_reasoning: bool = True
    default_reasoning_strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    max_reasoning_steps: int = 10
    reasoning_confidence_threshold: float = 0.75
    llm_model: str = "claude-3-opus"
    llm_api_key: Optional[str] = None

    # Ladybug settings
    enable_debugging: bool = True
    enable_tracing: bool = True
    enable_metrics: bool = True
    slow_query_threshold_ms: int = 100

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


class GraphQLAGI:
    """
    GraphQL AGI Client

    The main interface for GraphQL AGI functionality, providing:
    - GraphQL query execution with automatic Cypher translation
    - Semantic search using LanceDB vector storage
    - AGI reasoning and planning capabilities
    - Ladybug query analysis and debugging

    Example:
        >>> agi = GraphQLAGI("./my_database")
        >>>
        >>> # GraphQL query
        >>> result = agi.query('''
        ...     query {
        ...         users { id name }
        ...     }
        ... ''')
        >>>
        >>> # Semantic search
        >>> similar = agi.semantic_search("AI researcher", "users", top_k=5)
        >>>
        >>> # AGI reasoning
        >>> answer = agi.reason("What projects connect these users?")
    """

    def __init__(
        self,
        database_path: str = ":memory:",
        config: Optional[GraphQLAGIConfig] = None
    ):
        """
        Initialize GraphQL AGI client.

        Args:
            database_path: Path to Kuzu database (or ":memory:" for in-memory)
            config: Optional configuration object
        """
        self.config = config or GraphQLAGIConfig(database_path=database_path)
        self.config.database_path = database_path

        # Initialize Kuzu database
        self._db = kuzu.Database(
            database_path,
            buffer_pool_size=self.config.buffer_pool_size,
            read_only=self.config.read_only
        )
        self._conn = kuzu.Connection(self._db)

        # Load GraphQL AGI extension
        self._load_extension()

        # Initialize components
        self._vector_store: Optional[VectorStore] = None
        self._reasoning_engine: Optional[ReasoningEngine] = None
        self._planning_system: Optional[PlanningSystem] = None
        self._debugger: Optional[LadybugDebugger] = None

        self._schema_cache: Optional[str] = None

        logger.info(f"GraphQL AGI initialized with database: {database_path}")

    def _load_extension(self) -> None:
        """Load the GraphQL AGI extension into Kuzu."""
        try:
            # Try to load pre-built extension
            self._conn.execute("LOAD EXTENSION graphql_agi")
            logger.info("GraphQL AGI extension loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load extension: {e}. Using Python-only mode.")

    @property
    def vector_store(self) -> VectorStore:
        """Get or create the vector store instance."""
        if self._vector_store is None:
            vector_config = VectorStoreConfig(
                uri=self.config.lancedb_uri,
                dimension=self.config.vector_dimension,
                distance_metric=self.config.distance_metric,
                embedding_model=self.config.embedding_model
            )
            self._vector_store = VectorStore(vector_config)
        return self._vector_store

    @property
    def reasoning_engine(self) -> ReasoningEngine:
        """Get or create the reasoning engine instance."""
        if self._reasoning_engine is None:
            reasoning_config = ReasoningConfig(
                strategy=self.config.default_reasoning_strategy,
                max_steps=self.config.max_reasoning_steps,
                confidence_threshold=self.config.reasoning_confidence_threshold,
                llm_model=self.config.llm_model,
                api_key=self.config.llm_api_key
            )
            self._reasoning_engine = ReasoningEngine(
                config=reasoning_config,
                vector_store=self.vector_store,
                graph_connection=self._conn
            )
        return self._reasoning_engine

    @property
    def planning_system(self) -> PlanningSystem:
        """Get or create the planning system instance."""
        if self._planning_system is None:
            planning_config = PlanningConfig(
                max_plan_length=50,
                llm_model=self.config.llm_model,
                api_key=self.config.llm_api_key
            )
            self._planning_system = PlanningSystem(
                config=planning_config,
                reasoning_engine=self.reasoning_engine,
                graph_connection=self._conn
            )
        return self._planning_system

    @property
    def debugger(self) -> LadybugDebugger:
        """Get or create the Ladybug debugger instance."""
        if self._debugger is None:
            ladybug_config = LadybugConfig(
                enable_tracing=self.config.enable_tracing,
                enable_metrics=self.config.enable_metrics,
                slow_query_threshold_ms=self.config.slow_query_threshold_ms
            )
            self._debugger = LadybugDebugger(ladybug_config)
        return self._debugger

    # =========================================================================
    # GraphQL Operations
    # =========================================================================

    def query(
        self,
        graphql_query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query.

        Args:
            graphql_query: The GraphQL query string
            variables: Optional variables for the query
            operation_name: Optional operation name if query contains multiple operations

        Returns:
            Dictionary with 'data' and optionally 'errors' keys

        Example:
            >>> result = agi.query('''
            ...     query GetUsers($limit: Int!) {
            ...         users(first: $limit) {
            ...             id
            ...             name
            ...         }
            ...     }
            ... ''', variables={"limit": 10})
        """
        variables = variables or {}

        # Start debug session if enabled
        debug_session = None
        if self.config.enable_debugging:
            debug_session = self.debugger.start_session(graphql_query)

        try:
            # Parse and translate GraphQL to Cypher
            cypher_queries = self._translate_graphql(graphql_query, variables)

            # Execute queries
            results = {}
            for field_name, cypher in cypher_queries.items():
                result = self._conn.execute(cypher)
                results[field_name] = self._result_to_dict(result)

            response = {"data": results}

            # End debug session
            if debug_session:
                self.debugger.end_session(debug_session, success=True)

            return response

        except Exception as e:
            logger.error(f"GraphQL query failed: {e}")

            if debug_session:
                self.debugger.end_session(debug_session, success=False, error=str(e))

            return {
                "data": None,
                "errors": [{"message": str(e)}]
            }

    def mutate(
        self,
        mutation: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL mutation.

        Args:
            mutation: The GraphQL mutation string
            variables: Optional variables for the mutation

        Returns:
            Dictionary with 'data' and optionally 'errors' keys
        """
        return self.query(mutation, variables)

    def get_schema(self, refresh: bool = False) -> str:
        """
        Get the GraphQL schema in SDL format.

        Args:
            refresh: Whether to refresh the cached schema

        Returns:
            GraphQL schema as SDL string
        """
        if self._schema_cache is None or refresh:
            self._schema_cache = self._build_schema_sdl()
        return self._schema_cache

    def introspect(self) -> Dict[str, Any]:
        """
        Get full schema introspection result.

        Returns:
            Introspection result matching GraphQL introspection query format
        """
        return self.query("{ __schema { types { name } } }")

    # =========================================================================
    # Semantic Search Operations
    # =========================================================================

    def semantic_search(
        self,
        query: str,
        table: str = "*",
        top_k: int = 10,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across the knowledge graph.

        Args:
            query: Natural language search query
            table: Table/entity type to search (or "*" for all)
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0-1)
            filters: Additional filters to apply
            include_embeddings: Whether to include embedding vectors in results

        Returns:
            List of search results with id, score, type, and content

        Example:
            >>> results = agi.semantic_search(
            ...     query="machine learning researcher",
            ...     table="users",
            ...     top_k=5
            ... )
            >>> for r in results:
            ...     print(f"{r['id']}: {r['score']:.3f}")
        """
        # Generate embedding for query
        query_embedding = self.vector_store.embed(query)

        # Perform search
        results = self.vector_store.search(
            query_vector=query_embedding,
            table=table,
            top_k=top_k,
            threshold=threshold,
            filters=filters
        )

        # Format results
        formatted = []
        for result in results:
            item = {
                "id": result.entity_id,
                "score": result.score,
                "entityType": result.entity_type,
                "content": result.content,
                "metadata": result.metadata
            }
            if include_embeddings:
                item["embedding"] = result.vector
            formatted.append(item)

        return formatted

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
        Perform hybrid search combining vector similarity and keyword matching.

        Args:
            query: Search query
            table: Table to search
            vector_weight: Weight for vector similarity (0-1)
            text_weight: Weight for text matching (0-1)
            top_k: Number of results
            text_fields: Fields to search for text matching

        Returns:
            List of search results
        """
        return self.vector_store.hybrid_search(
            query=query,
            table=table,
            vector_weight=vector_weight,
            text_weight=text_weight,
            top_k=top_k,
            text_fields=text_fields
        )

    def store_embedding(
        self,
        entity_id: str,
        entity_type: str,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store an embedding for an entity.

        Args:
            entity_id: ID of the entity
            entity_type: Type of the entity
            text: Text to embed (if vector not provided)
            vector: Pre-computed embedding vector
            metadata: Additional metadata

        Returns:
            True if successful
        """
        if vector is None and text is not None:
            vector = self.vector_store.embed(text)
        elif vector is None:
            raise ValueError("Either text or vector must be provided")

        return self.vector_store.store(
            entity_id=entity_id,
            entity_type=entity_type,
            vector=vector,
            content=text,
            metadata=metadata
        )

    # =========================================================================
    # AGI Reasoning Operations
    # =========================================================================

    def reason(
        self,
        question: str,
        context: Optional[List[str]] = None,
        strategy: Optional[ReasoningStrategy] = None,
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform AGI reasoning to answer a question.

        Args:
            question: The question to answer
            context: Optional additional context
            strategy: Reasoning strategy to use
            max_steps: Maximum reasoning steps

        Returns:
            Reasoning result with answer, confidence, and trace

        Example:
            >>> result = agi.reason(
            ...     "What projects connect users in the AI field?",
            ...     strategy=ReasoningStrategy.TREE_OF_THOUGHTS
            ... )
            >>> print(f"Answer: {result['answer']}")
            >>> print(f"Confidence: {result['confidence']:.2%}")
        """
        result = self.reasoning_engine.reason(
            question=question,
            context=context or [],
            strategy=strategy or self.config.default_reasoning_strategy,
            max_steps=max_steps or self.config.max_reasoning_steps
        )

        return {
            "success": result.success,
            "answer": result.answer,
            "confidence": result.confidence,
            "steps": [
                {
                    "stepNumber": step.step_number,
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                    "confidence": step.confidence
                }
                for step in result.steps
            ],
            "relatedEntities": result.related_entities,
            "usedSources": result.used_sources
        }

    def plan(
        self,
        goal: str,
        constraints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create an execution plan to achieve a goal.

        Args:
            goal: The goal to achieve
            constraints: Optional constraints on the plan

        Returns:
            Plan with steps and execution order
        """
        result = self.planning_system.create_plan(
            goal=goal,
            constraints=constraints or []
        )

        return {
            "success": result.success,
            "planId": result.plan.plan_id if result.plan else None,
            "goal": goal,
            "steps": [
                {
                    "stepNumber": step.step_number,
                    "action": step.action_name,
                    "arguments": step.bound_arguments,
                    "description": step.description,
                    "dependsOn": step.depends_on
                }
                for step in (result.plan.steps if result.plan else [])
            ],
            "estimatedCost": result.plan.total_cost if result.plan else 0,
            "errors": result.failure_reasons
        }

    def multi_hop_reason(
        self,
        start_entity: str,
        question: str,
        max_hops: int = 3
    ) -> Dict[str, Any]:
        """
        Perform multi-hop reasoning starting from an entity.

        Args:
            start_entity: ID of the starting entity
            question: Question to answer
            max_hops: Maximum graph hops

        Returns:
            Multi-hop reasoning result
        """
        result = self.reasoning_engine.multi_hop_reason(
            start_entity=start_entity,
            question=question,
            max_hops=max_hops
        )

        return {
            "path": result.path,
            "answer": result.answer,
            "confidence": result.confidence,
            "steps": result.steps
        }

    # =========================================================================
    # Knowledge Graph Operations
    # =========================================================================

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity by ID."""
        query = f"MATCH (n) WHERE n.id = '{entity_id}' RETURN n"
        result = self._conn.execute(query)
        rows = list(result)
        if rows:
            return dict(rows[0])
        return None

    def get_neighbors(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None,
        hops: int = 1
    ) -> List[Dict[str, Any]]:
        """Get neighboring entities."""
        rel_pattern = ""
        if relationship_types:
            rel_pattern = ":" + "|".join(relationship_types)

        query = f"""
            MATCH (n)-[r{rel_pattern}*1..{hops}]-(m)
            WHERE n.id = '{entity_id}'
            RETURN DISTINCT m, type(r) as relationship
        """
        result = self._conn.execute(query)
        return [dict(row) for row in result]

    def get_subgraph(
        self,
        entity_id: str,
        radius: int = 2
    ) -> Dict[str, Any]:
        """Extract a subgraph around an entity."""
        query = f"""
            MATCH path = (n)-[*0..{radius}]-(m)
            WHERE n.id = '{entity_id}'
            RETURN path
        """
        result = self._conn.execute(query)

        entities = set()
        relationships = []

        for row in result:
            path = row[0]
            # Extract entities and relationships from path
            # (Implementation depends on Kuzu path format)

        return {
            "entities": list(entities),
            "relationships": relationships
        }

    # =========================================================================
    # Ladybug Debugging Operations
    # =========================================================================

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query using Ladybug.

        Args:
            query: GraphQL or Cypher query to analyze

        Returns:
            Analysis result with complexity, suggestions, etc.
        """
        analysis = self.debugger.analyze_query(query)

        return {
            "valid": analysis.valid,
            "complexity": analysis.complexity,
            "estimatedCost": analysis.estimated_cost,
            "warnings": analysis.warnings,
            "suggestions": analysis.suggestions,
            "queryPlan": analysis.query_plan
        }

    def get_query_metrics(self) -> Dict[str, Any]:
        """Get aggregated query metrics."""
        return self.debugger.get_metrics()

    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest queries."""
        return self.debugger.get_slow_queries(limit)

    def explain(self, query: str, analyze: bool = False) -> str:
        """
        Explain a query's execution plan.

        Args:
            query: Query to explain
            analyze: Whether to include actual execution stats

        Returns:
            Query plan as text
        """
        return self.debugger.explain_query(query, analyze=analyze)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _translate_graphql(
        self,
        query: str,
        variables: Dict[str, Any]
    ) -> Dict[str, str]:
        """Translate GraphQL query to Cypher queries."""
        # This is a simplified translation - full implementation in C++ extension
        # For now, use the extension if available, otherwise basic translation

        try:
            # Try using the extension
            result = self._conn.execute(
                f"CALL GRAPHQL_TRANSLATE('{query}', '{json.dumps(variables)}')"
            )
            translations = {}
            for row in result:
                translations[row[0]] = row[1]
            return translations
        except Exception:
            # Fall back to basic translation
            return self._basic_translate(query, variables)

    def _basic_translate(
        self,
        query: str,
        variables: Dict[str, Any]
    ) -> Dict[str, str]:
        """Basic GraphQL to Cypher translation."""
        # Very simplified translation for common patterns
        import re

        translations = {}

        # Match field selections
        pattern = r'(\w+)\s*(?:\([^)]*\))?\s*\{([^}]+)\}'
        matches = re.findall(pattern, query)

        for field_name, selections in matches:
            # Build basic MATCH ... RETURN query
            fields = [f.strip() for f in selections.split() if f.strip()]
            cypher = f"MATCH (n:{field_name}) RETURN " + ", ".join(
                f"n.{f}" for f in fields if f and not f.startswith('#')
            )
            translations[field_name] = cypher

        return translations

    def _result_to_dict(self, result) -> List[Dict[str, Any]]:
        """Convert Kuzu query result to list of dictionaries."""
        rows = []
        for row in result:
            if hasattr(row, '_asdict'):
                rows.append(row._asdict())
            elif hasattr(row, 'keys'):
                rows.append(dict(row))
            else:
                rows.append({"value": row})
        return rows

    def _build_schema_sdl(self) -> str:
        """Build GraphQL schema SDL from Kuzu catalog."""
        # Get table information from Kuzu
        schema_parts = ['schema {\n  query: Query\n  mutation: Mutation\n}\n']

        # Get node tables
        result = self._conn.execute("CALL show_tables() RETURN *")
        tables = list(result)

        for table in tables:
            table_name = table[0] if isinstance(table, tuple) else table.get('name', '')
            table_type = table[1] if isinstance(table, tuple) else table.get('type', '')

            if table_type == 'NODE':
                schema_parts.append(f'\ntype {table_name} implements Node {{\n')
                schema_parts.append('  id: ID!\n')

                # Get columns
                try:
                    col_result = self._conn.execute(
                        f"CALL table_info('{table_name}') RETURN *"
                    )
                    for col in col_result:
                        col_name = col[0] if isinstance(col, tuple) else col.get('name', '')
                        col_type = col[1] if isinstance(col, tuple) else col.get('type', '')
                        graphql_type = self._kuzu_to_graphql_type(col_type)
                        schema_parts.append(f'  {col_name}: {graphql_type}\n')
                except Exception:
                    pass

                schema_parts.append('}\n')

        return ''.join(schema_parts)

    def _kuzu_to_graphql_type(self, kuzu_type: str) -> str:
        """Convert Kuzu type to GraphQL type."""
        type_map = {
            'INT64': 'Int',
            'INT32': 'Int',
            'INT16': 'Int',
            'INT8': 'Int',
            'UINT64': 'Int',
            'UINT32': 'Int',
            'UINT16': 'Int',
            'UINT8': 'Int',
            'FLOAT': 'Float',
            'DOUBLE': 'Float',
            'STRING': 'String',
            'BOOL': 'Boolean',
            'DATE': 'DateTime',
            'TIMESTAMP': 'DateTime',
            'INTERVAL': 'String',
            'BLOB': 'String',
            'UUID': 'ID',
        }
        return type_map.get(kuzu_type.upper(), 'String')

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> GraphQLAGI:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the GraphQL AGI client and release resources."""
        if self._conn:
            self._conn = None
        if self._db:
            self._db = None
        if self._vector_store:
            self._vector_store.close()
        logger.info("GraphQL AGI client closed")
