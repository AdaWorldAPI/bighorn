"""
Ada AGI Surface - Unified wrapper for Kuzu + LanceDB + GraphQL + VSA + NARS

Endpoints:
    /agi/graph/*     -> Kuzu (Cypher queries)
    /agi/vector/*    -> LanceDB (similarity search)
    /agi/gql         -> GraphQL (flexible queries)
    /agi/self/*      -> Ada's self-model
    /agi/vsa/*       -> Vector Symbolic Architecture
    /agi/nars/*      -> NARS inference
    /health          -> Health check
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .kuzu_client import KuzuClient
from .lance_client import LanceClient
from .resolvers import create_schema
from .consumers import start_consumers
from .vsa import HypervectorSpace, CognitivePrimitives
from .nars import NARSReasoner

# =============================================================================
# CONFIGURATION
# =============================================================================

KUZU_DB_PATH = os.getenv("KUZU_DB_PATH", "/data/kuzu")
LANCE_DB_PATH = os.getenv("LANCE_DB_PATH", "/data/lancedb")
REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class CypherRequest(BaseModel):
    cypher: str
    params: Optional[Dict[str, Any]] = None


class VectorSearchRequest(BaseModel):
    vector: List[float]
    table: str = "thoughts"
    top_k: int = 10
    filter: Optional[Dict[str, Any]] = None


class VectorUpsertRequest(BaseModel):
    id: str
    vector: List[float]
    table: str = "thoughts"
    metadata: Optional[Dict[str, Any]] = None


class GraphQLRequest(BaseModel):
    query: str
    variables: Optional[Dict[str, Any]] = None
    operation_name: Optional[str] = None


class ThoughtRequest(BaseModel):
    content: str
    content_vector: List[float] = []
    style_33d: List[float] = []
    qualia_17d: List[float] = []
    parent_thought_id: Optional[str] = None
    session_id: Optional[str] = None
    confidence: float = 0.5
    importance: float = 0.5


class VSABindRequest(BaseModel):
    vectors: List[List[float]]


class VSASimilarityRequest(BaseModel):
    a: List[float]
    b: List[float]


class NARSInferRequest(BaseModel):
    premises: List[str]
    rule: str


class AdaptStyleRequest(BaseModel):
    style: Dict[str, Any]


class AdaptQualiaRequest(BaseModel):
    qualia: List[float]


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""

    # Initialize clients
    print(f"[AGI] Initializing Kuzu at {KUZU_DB_PATH}")
    app.state.kuzu = KuzuClient(KUZU_DB_PATH)

    print(f"[AGI] Initializing LanceDB at {LANCE_DB_PATH}")
    app.state.lance = LanceClient(LANCE_DB_PATH)

    print("[AGI] Initializing VSA space")
    app.state.vsa = HypervectorSpace()
    app.state.cognitive = CognitivePrimitives(app.state.vsa)

    print("[AGI] Initializing NARS reasoner")
    app.state.nars = NARSReasoner()

    # Initialize Kuzu schema and Observer
    await app.state.kuzu.init_observer()

    # Start Redis consumers if configured
    consumer_task = None
    if REDIS_URL and REDIS_TOKEN:
        print("[AGI] Starting Redis consumers")
        consumer_task = asyncio.create_task(
            start_consumers(
                app.state.kuzu,
                app.state.lance,
                REDIS_URL,
                REDIS_TOKEN,
            )
        )

    print("[AGI] Surface initialized")
    print(f"[AGI] Kuzu: {KUZU_DB_PATH}")
    print(f"[AGI] LanceDB: {LANCE_DB_PATH}")

    yield

    # Cleanup
    if consumer_task:
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

    app.state.kuzu.close()
    app.state.lance.close()
    print("[AGI] Surface shutdown")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Ada AGI Surface",
    description="Unified AGI interface: Kuzu + LanceDB + GraphQL + VSA + NARS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# KUZU ENDPOINTS
# =============================================================================

@app.post("/agi/graph/query")
async def graph_query(request: CypherRequest):
    """Execute Cypher query."""
    try:
        result = await app.state.kuzu.execute(request.cypher, request.params)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agi/graph/execute")
async def graph_execute(request: CypherRequest):
    """Execute Cypher mutation."""
    try:
        result = await app.state.kuzu.execute(request.cypher, request.params)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LANCEDB ENDPOINTS
# =============================================================================

@app.post("/agi/vector/search")
async def vector_search(request: VectorSearchRequest):
    """Similarity search in LanceDB."""
    try:
        results = await app.state.lance.search(
            request.vector,
            request.table,
            request.top_k,
            request.filter,
        )
        return {"ok": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agi/vector/upsert")
async def vector_upsert(request: VectorUpsertRequest):
    """Upsert vector to LanceDB."""
    try:
        await app.state.lance.upsert(
            request.id,
            request.vector,
            request.table,
            request.metadata,
        )
        return {"ok": True, "id": request.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GRAPHQL ENDPOINT
# =============================================================================

@app.post("/agi/gql")
async def graphql_endpoint(request: GraphQLRequest):
    """Execute GraphQL query."""
    try:
        from ariadne import graphql_sync

        schema = create_schema(
            app.state.kuzu,
            app.state.lance,
            app.state.vsa,
            app.state.nars,
        )

        success, result = graphql_sync(
            schema,
            {"query": request.query, "variables": request.variables},
            context_value={
                "kuzu": app.state.kuzu,
                "lance": app.state.lance,
                "vsa": app.state.vsa,
                "nars": app.state.nars,
            },
        )

        if success:
            return result
        else:
            raise HTTPException(status_code=400, detail=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agi/gql/schema")
async def graphql_schema():
    """Get GraphQL schema SDL."""
    from pathlib import Path
    schema_path = Path(__file__).parent / "schema.graphql"
    return {"schema": schema_path.read_text()}


# =============================================================================
# SELF-MODEL ENDPOINTS
# =============================================================================

@app.post("/agi/self/thought")
async def record_thought(request: ThoughtRequest):
    """
    Record a thought to Kuzu (graph) + LanceDB (vector).
    Called by hive.msgraph.de for each significant thought.
    """
    try:
        # Ensure default vectors
        style_vector = request.style_33d or ([0.33, 0.33, 0.34] + [0.11] * 9 + [0.2] * 5 + [0.0] * 16)
        qualia_vector = request.qualia_17d or [0.5] * 17

        # 1. Insert into Kuzu
        thought_id = await app.state.kuzu.create_thought(
            content=request.content,
            style_vector=style_vector,
            qualia_vector=qualia_vector,
            content_vector=request.content_vector,
            parent_id=request.parent_thought_id,
            session_id=request.session_id,
            confidence=request.confidence,
            importance=request.importance,
        )

        # 2. Index in LanceDB if content vector provided
        if request.content_vector:
            await app.state.lance.upsert(
                id=thought_id,
                vector=request.content_vector,
                table="thoughts",
                metadata={
                    "content": request.content[:500],
                    "session_id": request.session_id or "",
                },
            )

        return {"ok": True, "thought_id": thought_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agi/self/introspect")
async def introspect(query: str = "current_focus"):
    """
    Meta-cognition queries.

    Queries:
        - current_focus: What am I attending to?
        - recent_thoughts: Last N thoughts
        - reasoning_trace: How did I get here?
        - confidence: How certain am I?
        - emotional_state: Current qualia
        - cognitive_mode: Current thinking style
    """
    try:
        result = await app.state.kuzu.introspect(query)
        return {"ok": True, "query": query, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agi/self/trace")
async def reasoning_trace(depth: int = 10):
    """Get reasoning chain: Thought -> Thought -> ..."""
    try:
        trace = await app.state.kuzu.get_reasoning_trace(depth)
        return {"ok": True, "trace": trace}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agi/self/episodes")
async def episodic_memory(session_id: Optional[str] = None, limit: int = 20):
    """Query episodic memory."""
    try:
        episodes = await app.state.kuzu.get_episodes(session_id, limit)
        return {"ok": True, "episodes": episodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agi/self/adapt/style")
async def adapt_style(request: AdaptStyleRequest):
    """Adapt current thinking style."""
    try:
        await app.state.kuzu.update_observer_style(request.style)
        return {"ok": True, "style": request.style}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agi/self/adapt/qualia")
async def adapt_qualia(request: AdaptQualiaRequest):
    """Adapt current qualia state."""
    try:
        await app.state.kuzu.update_observer_qualia(request.qualia)
        return {"ok": True, "qualia": request.qualia}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# VSA ENDPOINTS
# =============================================================================

@app.post("/agi/vsa/bind")
async def vsa_bind(request: VSABindRequest):
    """Bind concepts using VSA XOR."""
    try:
        import numpy as np
        vectors = [np.array(v, dtype=np.int8) for v in request.vectors]
        result = app.state.vsa.bind_all(vectors)
        return {"ok": True, "vector": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agi/vsa/bundle")
async def vsa_bundle(request: VSABindRequest):
    """Bundle concepts using VSA majority vote."""
    try:
        import numpy as np
        vectors = [np.array(v, dtype=np.int8) for v in request.vectors]
        result = app.state.vsa.bundle(vectors)
        return {"ok": True, "vector": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agi/vsa/similarity")
async def vsa_similarity(request: VSASimilarityRequest):
    """Compute VSA similarity."""
    try:
        import numpy as np
        score = app.state.vsa.similarity(
            np.array(request.a, dtype=np.int8),
            np.array(request.b, dtype=np.int8),
        )
        return {"ok": True, "similarity": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agi/vsa/random")
async def vsa_random():
    """Generate random hypervector."""
    try:
        result = app.state.vsa.random()
        return {"ok": True, "vector": result.tolist(), "dimension": app.state.vsa.dimension}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# NARS ENDPOINTS
# =============================================================================

@app.post("/agi/nars/infer")
async def nars_infer(request: NARSInferRequest):
    """NARS inference step."""
    try:
        conclusion, truth = app.state.nars.infer(request.premises, request.rule)
        return {
            "ok": True,
            "conclusion": conclusion,
            "frequency": truth.frequency,
            "confidence": truth.confidence,
            "trace": app.state.nars.get_trace(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agi/nars/chain")
async def nars_chain(premises: List[str], max_steps: int = 10):
    """NARS multi-step inference chain."""
    try:
        conclusions = app.state.nars.chain_inference(premises, max_steps)
        return {
            "ok": True,
            "conclusions": [
                {"conclusion": c, "frequency": t.frequency, "confidence": t.confidence}
                for c, t in conclusions
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "kuzu": app.state.kuzu.is_connected(),
        "lance": app.state.lance.is_connected(),
        "vsa_dimension": app.state.vsa.dimension,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ada-agi-surface",
        "version": "1.0.0",
        "endpoints": {
            "graph": "/agi/graph/*",
            "vector": "/agi/vector/*",
            "gql": "/agi/gql",
            "self": "/agi/self/*",
            "vsa": "/agi/vsa/*",
            "nars": "/agi/nars/*",
        },
        "docs": "/docs",
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the server."""
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
