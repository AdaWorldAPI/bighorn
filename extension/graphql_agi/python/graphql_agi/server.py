"""
GraphQL AGI FastAPI Server

Provides a REST and GraphQL API for the GraphQL AGI system.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .core.client import GraphQLAGI, GraphQLAGIConfig
from .agi.reasoning import ReasoningStrategy

logger = logging.getLogger(__name__)

# Global client instance
_agi_client: Optional[GraphQLAGI] = None


def get_client() -> GraphQLAGI:
    """Get the global AGI client."""
    global _agi_client
    if _agi_client is None:
        raise RuntimeError("GraphQL AGI not initialized")
    return _agi_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _agi_client
    # Startup
    config = GraphQLAGIConfig()
    _agi_client = GraphQLAGI(config=config)
    logger.info("GraphQL AGI server started")
    yield
    # Shutdown
    if _agi_client:
        _agi_client.close()
    logger.info("GraphQL AGI server stopped")


# Create FastAPI app
app = FastAPI(
    title="GraphQL AGI",
    description="AI-native GraphQL interface for Kuzu graph database with LanceDB vector storage",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class GraphQLRequest(BaseModel):
    """GraphQL query request."""
    query: str = Field(..., description="GraphQL query string")
    variables: Optional[Dict[str, Any]] = Field(default=None, description="Query variables")
    operationName: Optional[str] = Field(default=None, description="Operation name")


class GraphQLResponse(BaseModel):
    """GraphQL query response."""
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None


class SemanticSearchRequest(BaseModel):
    """Semantic search request."""
    query: str = Field(..., description="Search query text")
    table: str = Field(default="*", description="Table to search")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Similarity threshold")


class SemanticSearchResult(BaseModel):
    """Single semantic search result."""
    id: str
    score: float
    entityType: str
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReasoningRequest(BaseModel):
    """AGI reasoning request."""
    question: str = Field(..., description="Question to reason about")
    context: Optional[List[str]] = Field(default=None, description="Additional context")
    strategy: Optional[str] = Field(default=None, description="Reasoning strategy")
    max_steps: Optional[int] = Field(default=None, ge=1, le=50, description="Max reasoning steps")


class ReasoningResponse(BaseModel):
    """AGI reasoning response."""
    success: bool
    answer: str
    confidence: float
    steps: List[Dict[str, Any]]
    relatedEntities: List[str]
    usedSources: List[str]


class PlanningRequest(BaseModel):
    """Planning request."""
    goal: str = Field(..., description="Goal to achieve")
    constraints: Optional[List[str]] = Field(default=None, description="Plan constraints")


class PlanningResponse(BaseModel):
    """Planning response."""
    success: bool
    planId: Optional[str] = None
    goal: str
    steps: List[Dict[str, Any]]
    estimatedCost: float
    errors: List[str]


class EmbeddingRequest(BaseModel):
    """Embedding storage request."""
    entityId: str = Field(..., description="Entity ID")
    entityType: str = Field(..., description="Entity type")
    text: Optional[str] = Field(default=None, description="Text to embed")
    vector: Optional[List[float]] = Field(default=None, description="Pre-computed vector")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class QueryAnalysisResponse(BaseModel):
    """Query analysis response."""
    valid: bool
    complexity: float
    estimatedCost: float
    warnings: List[str]
    suggestions: List[str]
    queryPlan: Optional[str] = None


# ============================================================================
# GraphQL Endpoint
# ============================================================================

@app.post("/graphql", response_model=GraphQLResponse, tags=["GraphQL"])
async def graphql_endpoint(request: GraphQLRequest):
    """
    Execute a GraphQL query.

    This endpoint accepts standard GraphQL queries and returns results.
    Queries are automatically translated to Cypher and executed against
    the underlying Kuzu graph database.
    """
    client = get_client()

    result = client.query(
        graphql_query=request.query,
        variables=request.variables,
        operation_name=request.operationName
    )

    return GraphQLResponse(**result)


@app.get("/graphql/schema", tags=["GraphQL"])
async def get_schema():
    """
    Get the GraphQL schema in SDL format.

    Returns the complete GraphQL schema including all types, queries,
    mutations, and AGI-specific extensions.
    """
    client = get_client()
    return Response(
        content=client.get_schema(),
        media_type="text/plain"
    )


# ============================================================================
# Semantic Search Endpoints
# ============================================================================

@app.post("/search", response_model=List[SemanticSearchResult], tags=["Search"])
async def semantic_search(request: SemanticSearchRequest):
    """
    Perform semantic search.

    Uses LanceDB vector similarity search to find entities that are
    semantically similar to the query text.
    """
    client = get_client()

    results = client.semantic_search(
        query=request.query,
        table=request.table,
        top_k=request.top_k,
        threshold=request.threshold
    )

    return [SemanticSearchResult(**r) for r in results]


@app.post("/search/hybrid", response_model=List[Dict[str, Any]], tags=["Search"])
async def hybrid_search(
    query: str,
    table: str,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    top_k: int = 10
):
    """
    Perform hybrid search.

    Combines vector similarity search with keyword matching for
    improved search relevance.
    """
    client = get_client()

    results = client.hybrid_search(
        query=query,
        table=table,
        vector_weight=vector_weight,
        text_weight=text_weight,
        top_k=top_k
    )

    return results


# ============================================================================
# AGI Reasoning Endpoints
# ============================================================================

@app.post("/reason", response_model=ReasoningResponse, tags=["AGI"])
async def reason(request: ReasoningRequest):
    """
    Perform AGI reasoning.

    Uses multi-step reasoning to answer complex questions by
    combining knowledge from the graph database and vector store.
    """
    client = get_client()

    # Parse strategy
    strategy = None
    if request.strategy:
        try:
            strategy = ReasoningStrategy(request.strategy)
        except ValueError:
            raise HTTPException(400, f"Invalid strategy: {request.strategy}")

    result = client.reason(
        question=request.question,
        context=request.context,
        strategy=strategy,
        max_steps=request.max_steps
    )

    return ReasoningResponse(**result)


@app.post("/plan", response_model=PlanningResponse, tags=["AGI"])
async def plan(request: PlanningRequest):
    """
    Create an execution plan.

    Generates a step-by-step plan to achieve the specified goal,
    using available actions and respecting constraints.
    """
    client = get_client()

    result = client.plan(
        goal=request.goal,
        constraints=request.constraints
    )

    return PlanningResponse(**result)


@app.post("/multihop", tags=["AGI"])
async def multi_hop_reason(
    start_entity: str,
    question: str,
    max_hops: int = 3
):
    """
    Perform multi-hop reasoning.

    Traverses the knowledge graph starting from a specific entity
    to answer questions that require multiple reasoning steps.
    """
    client = get_client()

    result = client.multi_hop_reason(
        start_entity=start_entity,
        question=question,
        max_hops=max_hops
    )

    return result


# ============================================================================
# Vector Store Endpoints
# ============================================================================

@app.post("/embeddings", tags=["Vectors"])
async def store_embedding(request: EmbeddingRequest):
    """
    Store an embedding for an entity.

    Either provide text to be embedded or a pre-computed vector.
    """
    client = get_client()

    success = client.store_embedding(
        entity_id=request.entityId,
        entity_type=request.entityType,
        text=request.text,
        vector=request.vector,
        metadata=request.metadata
    )

    if not success:
        raise HTTPException(500, "Failed to store embedding")

    return {"success": True, "entityId": request.entityId}


@app.post("/embeddings/batch", tags=["Vectors"])
async def store_embeddings_batch(items: List[EmbeddingRequest]):
    """
    Store multiple embeddings in batch.
    """
    client = get_client()
    count = 0

    for item in items:
        success = client.store_embedding(
            entity_id=item.entityId,
            entity_type=item.entityType,
            text=item.text,
            vector=item.vector,
            metadata=item.metadata
        )
        if success:
            count += 1

    return {"success": True, "stored": count, "total": len(items)}


# ============================================================================
# Ladybug Debug Endpoints
# ============================================================================

@app.post("/debug/analyze", response_model=QueryAnalysisResponse, tags=["Debug"])
async def analyze_query(query: str):
    """
    Analyze a query using Ladybug.

    Returns complexity metrics, warnings, and optimization suggestions.
    """
    client = get_client()

    result = client.analyze_query(query)

    return QueryAnalysisResponse(**result)


@app.get("/debug/metrics", tags=["Debug"])
async def get_metrics():
    """
    Get aggregated query metrics.

    Returns performance statistics across all executed queries.
    """
    client = get_client()
    return client.get_query_metrics()


@app.get("/debug/slow-queries", tags=["Debug"])
async def get_slow_queries(limit: int = 10):
    """
    Get the slowest queries.

    Useful for identifying performance bottlenecks.
    """
    client = get_client()
    return client.get_slow_queries(limit)


@app.post("/debug/explain", tags=["Debug"])
async def explain_query(query: str, analyze: bool = False):
    """
    Explain a query's execution plan.

    If analyze=True, actually executes the query and includes
    real execution statistics.
    """
    client = get_client()
    plan = client.explain(query, analyze=analyze)

    return Response(content=plan, media_type="text/plain")


# ============================================================================
# Knowledge Graph Endpoints
# ============================================================================

@app.get("/entities/{entity_id}", tags=["Graph"])
async def get_entity(entity_id: str):
    """
    Get an entity by ID.
    """
    client = get_client()
    entity = client.get_entity(entity_id)

    if not entity:
        raise HTTPException(404, f"Entity not found: {entity_id}")

    return entity


@app.get("/entities/{entity_id}/neighbors", tags=["Graph"])
async def get_neighbors(
    entity_id: str,
    relationship_types: Optional[str] = None,
    hops: int = 1
):
    """
    Get neighboring entities.
    """
    client = get_client()

    rel_types = relationship_types.split(",") if relationship_types else None

    neighbors = client.get_neighbors(
        entity_id=entity_id,
        relationship_types=rel_types,
        hops=hops
    )

    return neighbors


@app.get("/entities/{entity_id}/subgraph", tags=["Graph"])
async def get_subgraph(entity_id: str, radius: int = 2):
    """
    Extract a subgraph around an entity.
    """
    client = get_client()

    subgraph = client.get_subgraph(
        entity_id=entity_id,
        radius=radius
    )

    return subgraph


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/", response_class=HTMLResponse, tags=["System"])
async def root():
    """Root endpoint with API documentation."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>GraphQL AGI</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #2196F3; }
        .card { background: #f5f5f5; border-radius: 8px; padding: 20px; margin: 20px 0; }
        a { color: #2196F3; }
        code { background: #e0e0e0; padding: 2px 6px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>GraphQL AGI</h1>
    <p>AI-native GraphQL interface for Kuzu graph database with LanceDB vector storage.</p>

    <div class="card">
        <h3>Quick Links</h3>
        <ul>
            <li><a href="/docs">Interactive API Documentation (Swagger)</a></li>
            <li><a href="/redoc">Alternative API Documentation (ReDoc)</a></li>
            <li><a href="/graphql/schema">GraphQL Schema (SDL)</a></li>
            <li><a href="/health">Health Check</a></li>
        </ul>
    </div>

    <div class="card">
        <h3>Features</h3>
        <ul>
            <li><strong>GraphQL API</strong> - Query your graph with intuitive GraphQL</li>
            <li><strong>Semantic Search</strong> - Vector similarity search with LanceDB</li>
            <li><strong>AGI Reasoning</strong> - Multi-step reasoning with Chain-of-Thought</li>
            <li><strong>Planning</strong> - Goal-oriented task planning</li>
            <li><strong>Ladybug Debug</strong> - Query analysis and performance monitoring</li>
        </ul>
    </div>

    <div class="card">
        <h3>Example Usage</h3>
        <pre><code>
# GraphQL Query
curl -X POST http://localhost:8000/graphql \\
  -H "Content-Type: application/json" \\
  -d '{"query": "{ users { id name } }"}'

# Semantic Search
curl -X POST http://localhost:8000/search \\
  -H "Content-Type: application/json" \\
  -d '{"query": "AI researcher", "table": "users", "top_k": 5}'

# AGI Reasoning
curl -X POST http://localhost:8000/reason \\
  -H "Content-Type: application/json" \\
  -d '{"question": "What connects these users?"}'
        </code></pre>
    </div>
</body>
</html>
"""


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Run the GraphQL AGI server."""
    import uvicorn

    uvicorn.run(
        "graphql_agi.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
