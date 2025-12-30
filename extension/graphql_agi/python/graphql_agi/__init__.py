"""
GraphQL AGI - A Perfect Integration of GraphQL, LanceDB, AGI Reasoning, and Ladybug Debugging

This package provides a comprehensive AI-native graph database interface combining:
- GraphQL API for intuitive query interface
- LanceDB for vector storage and semantic search
- AGI reasoning engine for intelligent query processing
- Ladybug debugging for query analysis and visualization

Example:
    >>> from graphql_agi import GraphQLAGI
    >>>
    >>> # Initialize with a Kuzu database
    >>> agi = GraphQLAGI("./my_database")
    >>>
    >>> # Execute a GraphQL query
    >>> result = agi.query('''
    ...     query {
    ...         users(first: 10) {
    ...             id
    ...             name
    ...             email
    ...         }
    ...     }
    ... ''')
    >>>
    >>> # Perform semantic search
    >>> similar = agi.semantic_search(
    ...     query="machine learning expert",
    ...     table="users",
    ...     top_k=5
    ... )
    >>>
    >>> # Use AGI reasoning
    >>> answer = agi.reason(
    ...     question="What users are connected to AI research projects?",
    ...     strategy="chain_of_thought"
    ... )
"""

__version__ = "1.0.0"
__author__ = "GraphQL AGI Team"
__license__ = "MIT"

from .core.client import GraphQLAGI, GraphQLAGIConfig
from .core.schema import SchemaBuilder, GraphQLType, GraphQLField
from .lancedb.vector_store import VectorStore, VectorSearchResult
from .agi.reasoning import ReasoningEngine, ReasoningResult, ReasoningStrategy
from .agi.planning import PlanningSystem, Plan, PlanStep
from .ladybug.debugger import LadybugDebugger, QueryAnalysis, QueryMetrics

__all__ = [
    # Core
    "GraphQLAGI",
    "GraphQLAGIConfig",
    "SchemaBuilder",
    "GraphQLType",
    "GraphQLField",
    # Vector Store
    "VectorStore",
    "VectorSearchResult",
    # AGI
    "ReasoningEngine",
    "ReasoningResult",
    "ReasoningStrategy",
    "PlanningSystem",
    "Plan",
    "PlanStep",
    # Ladybug
    "LadybugDebugger",
    "QueryAnalysis",
    "QueryMetrics",
]
