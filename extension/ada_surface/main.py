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
from .thinking_styles import ResonanceEngine, STYLES, get_style, all_styles, RI

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


class TextureRequest(BaseModel):
    """9-channel texture for style emergence."""
    tension: float = 0.5
    novelty: float = 0.5
    intimacy: float = 0.5
    clarity: float = 0.5
    urgency: float = 0.5
    depth: float = 0.5
    play: float = 0.5
    stability: float = 0.5
    abstraction: float = 0.5


class StyleSearchRequest(BaseModel):
    """Search styles by vector similarity."""
    vector: List[float]
    top_k: int = 5
    category: Optional[str] = None
    tier: Optional[int] = None


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

    print("[AGI] Initializing Resonance Engine")
    app.state.resonance = ResonanceEngine()

    # Initialize Kuzu schema and Observer
    await app.state.kuzu.init_observer()

    # Index thinking styles in LanceDB
    print("[AGI] Indexing 36 thinking styles")
    style_count = await app.state.lance.index_styles()
    print(f"[AGI] Indexed {style_count} thinking styles")

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
    """
    Bind concepts using VSA XOR.
    
    Input vectors can be:
    - Full 10K bipolar vectors [-1, 1]
    - Smaller vectors (will be converted to bipolar and padded)
    - Float vectors (thresholded at 0)
    """
    try:
        import numpy as np
        
        def to_bipolar(v):
            arr = np.array(v, dtype=np.float32)
            # If already bipolar and full size, use as-is
            if len(arr) == app.state.vsa.dimension and set(np.unique(arr)).issubset({-1, 0, 1}):
                arr[arr == 0] = np.random.choice([-1, 1], size=np.sum(arr == 0))
                return arr.astype(np.int8)
            # Otherwise, threshold at 0 and pad to dimension
            bipolar = np.sign(arr)
            bipolar[bipolar == 0] = np.random.choice([-1, 1], size=np.sum(bipolar == 0))
            # Pad or truncate to dimension
            if len(bipolar) < app.state.vsa.dimension:
                # Pad with deterministic random based on input hash
                seed = hash(tuple(v)) % (2**32)
                rng = np.random.default_rng(seed)
                padding = rng.choice([-1, 1], size=app.state.vsa.dimension - len(bipolar))
                bipolar = np.concatenate([bipolar, padding])
            return bipolar[:app.state.vsa.dimension].astype(np.int8)
        
        vectors = [to_bipolar(v) for v in request.vectors]
        result = app.state.vsa.bind_all(vectors)
        return {"ok": True, "vector": result.tolist(), "dimension": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agi/vsa/bundle")
async def vsa_bundle(request: VSABindRequest):
    """
    Bundle concepts using VSA majority vote.
    
    Input vectors can be any size/format - will be converted to bipolar.
    """
    try:
        import numpy as np
        
        def to_bipolar(v):
            arr = np.array(v, dtype=np.float32)
            if len(arr) == app.state.vsa.dimension and set(np.unique(arr)).issubset({-1, 0, 1}):
                arr[arr == 0] = np.random.choice([-1, 1], size=np.sum(arr == 0))
                return arr.astype(np.int8)
            bipolar = np.sign(arr)
            bipolar[bipolar == 0] = np.random.choice([-1, 1], size=np.sum(bipolar == 0))
            if len(bipolar) < app.state.vsa.dimension:
                seed = hash(tuple(v)) % (2**32)
                rng = np.random.default_rng(seed)
                padding = rng.choice([-1, 1], size=app.state.vsa.dimension - len(bipolar))
                bipolar = np.concatenate([bipolar, padding])
            return bipolar[:app.state.vsa.dimension].astype(np.int8)
        
        vectors = [to_bipolar(v) for v in request.vectors]
        result = app.state.vsa.bundle(vectors)
        return {"ok": True, "vector": result.tolist(), "dimension": len(result)}
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
# THINKING STYLES ENDPOINTS
# =============================================================================

@app.get("/agi/styles")
async def list_styles():
    """List all 36 thinking styles."""
    try:
        styles = all_styles()
        return {
            "ok": True,
            "count": len(styles),
            "styles": [
                {
                    "id": s.id,
                    "name": s.name,
                    "category": s.category.value,
                    "tier": s.tier.value,
                    "description": s.description,
                    "microcode": s.microcode,
                }
                for s in styles
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agi/styles/{style_id}")
async def get_style_by_id(style_id: str):
    """Get a specific thinking style by ID."""
    try:
        style = get_style(style_id)
        if not style:
            raise HTTPException(status_code=404, detail=f"Style not found: {style_id}")
        return {
            "ok": True,
            "style": {
                "id": style.id,
                "name": style.name,
                "category": style.category.value,
                "tier": style.tier.value,
                "description": style.description,
                "microcode": style.microcode,
                "resonance": {k.value: v for k, v in style.resonance.items()},
                "glyph": style.glyph,
                "chains_to": style.chains_to,
                "chains_from": style.chains_from,
                "min_rung": style.min_rung,
                "max_rung": style.max_rung,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agi/styles/emerge")
async def emerge_style(request: TextureRequest):
    """
    Emerge the best-fit thinking style from a 9-channel texture.

    The resonance engine computes compatibility between the input texture
    and each style's resonance profile, returning the top matches.
    """
    try:
        # Build texture dict from request
        texture = {
            RI.TENSION: request.tension,
            RI.NOVELTY: request.novelty,
            RI.INTIMACY: request.intimacy,
            RI.CLARITY: request.clarity,
            RI.URGENCY: request.urgency,
            RI.DEPTH: request.depth,
            RI.PLAY: request.play,
            RI.STABILITY: request.stability,
            RI.ABSTRACTION: request.abstraction,
        }

        # Get emerged styles
        emerged = app.state.resonance.emerge(texture, top_k=5)

        return {
            "ok": True,
            "texture": {k.value: v for k, v in texture.items()},
            "emerged": [
                {
                    "style_id": style.id,
                    "name": style.name,
                    "category": style.category.value,
                    "tier": style.tier.value,
                    "score": score,
                    "microcode": style.microcode,
                }
                for style, score in emerged
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agi/styles/search")
async def search_styles(request: StyleSearchRequest):
    """Search thinking styles by vector similarity in LanceDB."""
    try:
        results = await app.state.lance.search_styles(
            vector=request.vector,
            top_k=request.top_k,
            category=request.category,
            tier=request.tier,
        )
        return {"ok": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agi/styles/categories")
async def list_categories():
    """List all style categories with their styles."""
    try:
        from .thinking_styles import StyleCategory

        categories = {}
        for style in all_styles():
            cat = style.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "id": style.id,
                "name": style.name,
                "tier": style.tier.value,
            })

        return {
            "ok": True,
            "categories": categories,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agi/styles/chains/{style_id}")
async def get_style_chains(style_id: str):
    """Get style transition chains for a given style."""
    try:
        style = get_style(style_id)
        if not style:
            raise HTTPException(status_code=404, detail=f"Style not found: {style_id}")

        # Resolve chain references to full style objects
        chains_to = [get_style(sid) for sid in style.chains_to]
        chains_from = [get_style(sid) for sid in style.chains_from]

        return {
            "ok": True,
            "style_id": style_id,
            "chains_to": [
                {"id": s.id, "name": s.name, "microcode": s.microcode}
                for s in chains_to if s
            ],
            "chains_from": [
                {"id": s.id, "name": s.name, "microcode": s.microcode}
                for s in chains_from if s
            ],
        }
    except HTTPException:
        raise
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
        "version": "1.1.0",
        "endpoints": {
            "graph": "/agi/graph/*",
            "vector": "/agi/vector/*",
            "gql": "/agi/gql",
            "self": "/agi/self/*",
            "vsa": "/agi/vsa/*",
            "nars": "/agi/nars/*",
            "styles": "/agi/styles/*",
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

