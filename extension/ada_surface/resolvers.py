"""
GraphQL Resolvers for AGI Stack.

Connects GraphQL schema to Kuzu and LanceDB backends.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from ariadne import (
    QueryType,
    MutationType,
    ObjectType,
    make_executable_schema,
    ScalarType,
)

from .kuzu_client import KuzuClient
from .lance_client import LanceClient
from .vsa import HypervectorSpace
from .nars import NARSReasoner, TruthValue

# =============================================================================
# SCALAR TYPES
# =============================================================================

datetime_scalar = ScalarType("DateTime")
json_scalar = ScalarType("JSON")


@datetime_scalar.serializer
def serialize_datetime(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@datetime_scalar.value_parser
def parse_datetime_value(value):
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return value


@json_scalar.serializer
def serialize_json(value):
    return value


@json_scalar.value_parser
def parse_json_value(value):
    return value


# =============================================================================
# TYPE RESOLVERS
# =============================================================================

query = QueryType()
mutation = MutationType()
observer_type = ObjectType("Observer")
thought_type = ObjectType("Thought")
episode_type = ObjectType("Episode")
thinking_style_type = ObjectType("ThinkingStyle")
qualia_type = ObjectType("Qualia")

# Style presets
STYLE_PRESETS = {
    "HYBRID": "Balanced cognitive mode for general tasks",
    "EMPATHIC": "Relational, empathic mode for personal connection",
    "WORK": "Analytical, task-oriented mode",
    "CREATIVE": "Creative, exploratory, boundary-pushing mode",
    "AGI": "Deep meta-cognitive mode for reflection",
}

# Qualia families
QUALIA_FAMILIES = {
    "emberglow": "Warm, connected, present",
    "woodwarm": "Grounded, stable, nurturing",
    "steelwind": "Sharp, clear, precise",
    "oceandrift": "Flowing, receptive, deep",
    "frostbite": "Crisp, boundaried, analytical",
    "sunburst": "Radiant, energetic, expressive",
    "nightshade": "Mysterious, contemplative, shadowed",
    "thornrose": "Beautiful tension, bittersweet",
    "velvetdusk": "Soft, twilight, transitional",
    "stormbreak": "Intense, cathartic, releasing",
}


# =============================================================================
# QUERY RESOLVERS
# =============================================================================

@query.field("self")
async def resolve_self(_, info):
    """Get Ada's self-model (Observer)."""
    kuzu: KuzuClient = info.context["kuzu"]

    result = await kuzu.execute("""
        MATCH (o:Observer {id: 'ada'})
        RETURN o.id AS id, o.name AS name,
               o.current_goal AS currentGoal,
               o.confidence AS confidence,
               o.style_vector AS styleVector,
               o.qualia_vector AS qualiaVector
    """)

    if not result:
        # Create default observer if not exists
        return {
            "id": "ada",
            "name": "Ada",
            "currentGoal": "Be present and helpful",
            "confidence": 0.5,
            "styleVector": [],
            "qualiaVector": [],
        }

    return result[0]


@query.field("thought")
async def resolve_thought(_, info, id: str):
    """Get thought by ID."""
    kuzu: KuzuClient = info.context["kuzu"]

    result = await kuzu.execute("""
        MATCH (t:Thought {id: $id})
        RETURN t.id AS id, t.content AS content,
               t.timestamp AS timestamp, t.session_id AS sessionId,
               t.step_number AS stepNumber, t.confidence AS confidence,
               t.importance AS importance, t.style_vector AS styleVector,
               t.qualia_vector AS qualiaVector
    """, {"id": id})

    return result[0] if result else None


@query.field("thoughts")
async def resolve_thoughts(_, info, sessionId: str = None, limit: int = 20, offset: int = 0):
    """Get thoughts with optional filtering."""
    kuzu: KuzuClient = info.context["kuzu"]

    if sessionId:
        return await kuzu.execute("""
            MATCH (t:Thought {session_id: $sessionId})
            RETURN t.id AS id, t.content AS content,
                   t.timestamp AS timestamp, t.session_id AS sessionId,
                   t.step_number AS stepNumber, t.confidence AS confidence,
                   t.importance AS importance
            ORDER BY t.timestamp DESC
            LIMIT $limit SKIP $offset
        """, {"sessionId": sessionId, "limit": limit, "offset": offset})
    else:
        return await kuzu.execute("""
            MATCH (t:Thought)
            RETURN t.id AS id, t.content AS content,
                   t.timestamp AS timestamp, t.session_id AS sessionId,
                   t.step_number AS stepNumber, t.confidence AS confidence,
                   t.importance AS importance
            ORDER BY t.timestamp DESC
            LIMIT $limit SKIP $offset
        """, {"limit": limit, "offset": offset})


@query.field("similarThoughts")
async def resolve_similar_thoughts(_, info, vector: List[float], topK: int = 10):
    """Find similar thoughts by vector."""
    lance: LanceClient = info.context["lance"]

    results = await lance.search(vector, "thoughts", topK)

    # Convert to thought format
    return [
        {
            "id": r.get("id", ""),
            "content": r.get("content", ""),
            "sessionId": r.get("session_id", ""),
            "confidence": r.get("confidence", 0.5),
            "stepNumber": 0,
            "importance": 0.5,
            "timestamp": r.get("timestamp", datetime.utcnow().isoformat()),
        }
        for r in results
    ]


@query.field("episode")
async def resolve_episode(_, info, id: str):
    """Get episode by ID."""
    kuzu: KuzuClient = info.context["kuzu"]

    result = await kuzu.execute("""
        MATCH (e:Episode {id: $id})
        RETURN e.id AS id, e.session_id AS sessionId,
               e.summary AS summary, e.start_time AS startTime,
               e.end_time AS endTime, e.emotional_valence AS emotionalValence,
               e.importance AS importance
    """, {"id": id})

    return result[0] if result else None


@query.field("episodes")
async def resolve_episodes(_, info, sessionId: str = None, limit: int = 10):
    """Get episodes with optional filtering."""
    kuzu: KuzuClient = info.context["kuzu"]

    if sessionId:
        return await kuzu.execute("""
            MATCH (e:Episode {session_id: $sessionId})
            RETURN e.id AS id, e.session_id AS sessionId,
                   e.summary AS summary, e.start_time AS startTime,
                   e.emotional_valence AS emotionalValence,
                   e.importance AS importance
            ORDER BY e.start_time DESC
            LIMIT $limit
        """, {"sessionId": sessionId, "limit": limit})
    else:
        return await kuzu.execute("""
            MATCH (e:Episode)
            RETURN e.id AS id, e.session_id AS sessionId,
                   e.summary AS summary, e.start_time AS startTime,
                   e.emotional_valence AS emotionalValence,
                   e.importance AS importance
            ORDER BY e.start_time DESC
            LIMIT $limit
        """, {"limit": limit})


@query.field("concept")
async def resolve_concept(_, info, id: str):
    """Get concept by ID."""
    kuzu: KuzuClient = info.context["kuzu"]

    result = await kuzu.execute("""
        MATCH (c:Concept {id: $id})
        RETURN c.id AS id, c.name AS name,
               c.salience AS salience, c.activation AS activation,
               c.created_at AS createdAt, c.accessed_at AS accessedAt
    """, {"id": id})

    return result[0] if result else None


@query.field("concepts")
async def resolve_concepts(_, info, query: str = None, limit: int = 20):
    """Get concepts."""
    kuzu: KuzuClient = info.context["kuzu"]

    # For now, just return all concepts ordered by salience
    return await kuzu.execute("""
        MATCH (c:Concept)
        RETURN c.id AS id, c.name AS name,
               c.salience AS salience, c.activation AS activation,
               c.created_at AS createdAt, c.accessed_at AS accessedAt
        ORDER BY c.salience DESC
        LIMIT $limit
    """, {"limit": limit})


@query.field("reasoningTrace")
async def resolve_reasoning_trace(_, info, fromThoughtId: str = None, depth: int = 10):
    """Get reasoning chain."""
    kuzu: KuzuClient = info.context["kuzu"]
    return await kuzu.get_reasoning_trace(depth)


@query.field("introspect")
async def resolve_introspect(_, info, query: str):
    """Meta-cognition queries."""
    kuzu: KuzuClient = info.context["kuzu"]
    return await kuzu.introspect(query)


@query.field("stylePresets")
async def resolve_style_presets(_, info):
    """Get available style presets."""
    return [
        {"name": name, "description": desc}
        for name, desc in STYLE_PRESETS.items()
    ]


@query.field("qualiaFamilies")
async def resolve_qualia_families(_, info):
    """Get available qualia families."""
    return [
        {"name": name, "description": desc, "variantCount": 87}
        for name, desc in QUALIA_FAMILIES.items()
    ]


# =============================================================================
# MUTATION RESOLVERS
# =============================================================================

@mutation.field("createThought")
async def resolve_create_thought(_, info, input: Dict):
    """Create a new thought."""
    kuzu: KuzuClient = info.context["kuzu"]
    lance: LanceClient = info.context["lance"]

    # Extract style and qualia vectors
    style_vector = []
    qualia_vector = []

    if input.get("style"):
        style_vector = input["style"].get("dense", [0.0] * 33)
    if input.get("qualia"):
        qualia_vector = input["qualia"].get("vector", [0.5] * 17)

    # Create thought in Kuzu
    thought_id = await kuzu.create_thought(
        content=input["content"],
        style_vector=style_vector,
        qualia_vector=qualia_vector,
        content_vector=input.get("contentVector", []),
        parent_id=input.get("parentThoughtId"),
        session_id=input.get("sessionId"),
        confidence=input.get("confidence", 0.5),
        importance=input.get("importance", 0.5),
    )

    # Index in LanceDB if content vector provided
    if input.get("contentVector"):
        await lance.upsert(
            id=thought_id,
            vector=input["contentVector"],
            table="thoughts",
            metadata={
                "content": input["content"][:500],
                "session_id": input.get("sessionId", ""),
            }
        )

    return {
        "id": thought_id,
        "content": input["content"],
        "timestamp": datetime.utcnow().isoformat(),
        "sessionId": input.get("sessionId"),
        "stepNumber": 0,
        "confidence": input.get("confidence", 0.5),
        "importance": input.get("importance", 0.5),
    }


@mutation.field("linkThoughts")
async def resolve_link_thoughts(_, info, fromId: str, toId: str):
    """Link two thoughts."""
    kuzu: KuzuClient = info.context["kuzu"]
    return await kuzu.link_thoughts(fromId, toId)


@mutation.field("createEpisode")
async def resolve_create_episode(_, info, input: Dict):
    """Create a new episode."""
    kuzu: KuzuClient = info.context["kuzu"]

    episode_id = await kuzu.create_episode(
        session_id=input["sessionId"],
        summary=input.get("summary", ""),
        thought_ids=input.get("thoughtIds", []),
    )

    return {
        "id": episode_id,
        "sessionId": input["sessionId"],
        "summary": input.get("summary", ""),
        "startTime": datetime.utcnow().isoformat(),
        "emotionalValence": 0.5,
        "importance": 0.5,
        "thoughtCount": len(input.get("thoughtIds", [])),
    }


@mutation.field("closeEpisode")
async def resolve_close_episode(_, info, id: str, summary: str = None):
    """Close an episode."""
    kuzu: KuzuClient = info.context["kuzu"]

    await kuzu.execute("""
        MATCH (e:Episode {id: $id})
        SET e.end_time = timestamp(),
            e.summary = $summary
    """, {"id": id, "summary": summary or ""})

    result = await kuzu.execute("""
        MATCH (e:Episode {id: $id})
        RETURN e.id AS id, e.session_id AS sessionId,
               e.summary AS summary, e.start_time AS startTime,
               e.end_time AS endTime
    """, {"id": id})

    return result[0] if result else None


@mutation.field("updateGoal")
async def resolve_update_goal(_, info, goal: str = None):
    """Update Observer's current goal."""
    kuzu: KuzuClient = info.context["kuzu"]

    await kuzu.execute("""
        MATCH (o:Observer {id: 'ada'})
        SET o.current_goal = $goal,
            o.updated_at = timestamp()
    """, {"goal": goal or ""})

    return await resolve_self(_, info)


@mutation.field("updateStyle")
async def resolve_update_style(_, info, style: Dict):
    """Update Observer's thinking style."""
    kuzu: KuzuClient = info.context["kuzu"]
    await kuzu.update_observer_style(style)
    return await resolve_self(_, info)


@mutation.field("updateQualia")
async def resolve_update_qualia(_, info, qualia: Dict):
    """Update Observer's qualia state."""
    kuzu: KuzuClient = info.context["kuzu"]

    # Convert qualia input to vector
    qualia_vector = [
        qualia.get("arousal", 0.5),
        qualia.get("valence", 0.5),
        qualia.get("tension", 0.5),
        qualia.get("warmth", 0.5),
        qualia.get("clarity", 0.5),
        qualia.get("boundary", 0.5),
        qualia.get("depth", 0.5),
        qualia.get("velocity", 0.5),
        qualia.get("entropy", 0.5),
        qualia.get("coherence", 0.5),
        qualia.get("intimacy", 0.5),
        qualia.get("presence", 0.5),
        qualia.get("assertion", 0.5),
        qualia.get("receptivity", 0.5),
        qualia.get("groundedness", 0.5),
        qualia.get("expansion", 0.5),
        qualia.get("integration", 0.5),
    ]

    await kuzu.update_observer_qualia(qualia_vector)
    return await resolve_self(_, info)


@mutation.field("createConcept")
async def resolve_create_concept(_, info, name: str, vector: List[float] = None):
    """Create a new concept."""
    kuzu: KuzuClient = info.context["kuzu"]

    concept_id = await kuzu.create_concept(
        name=name,
        content_vector=vector,
    )

    return {
        "id": concept_id,
        "name": name,
        "salience": 0.5,
        "activation": 0.0,
        "createdAt": datetime.utcnow().isoformat(),
        "accessedAt": datetime.utcnow().isoformat(),
    }


@mutation.field("linkConcepts")
async def resolve_link_concepts(_, info, fromId: str, toId: str, strength: float = 0.5):
    """Link two concepts."""
    kuzu: KuzuClient = info.context["kuzu"]
    return await kuzu.link_concepts(fromId, toId, strength=strength)


@mutation.field("vsaBind")
async def resolve_vsa_bind(_, info, vectors: List[List[float]]):
    """Bind vectors using VSA XOR."""
    vsa: HypervectorSpace = info.context["vsa"]

    import numpy as np
    np_vectors = [np.array(v, dtype=np.int8) for v in vectors]
    result = vsa.bind_all(np_vectors)

    return result.tolist()


@mutation.field("vsaBundle")
async def resolve_vsa_bundle(_, info, vectors: List[List[float]]):
    """Bundle vectors using VSA majority vote."""
    vsa: HypervectorSpace = info.context["vsa"]

    import numpy as np
    np_vectors = [np.array(v, dtype=np.int8) for v in vectors]
    result = vsa.bundle(np_vectors)

    return result.tolist()


@mutation.field("infer")
async def resolve_infer(_, info, premises: List[str], rule: str):
    """NARS inference step."""
    nars: NARSReasoner = info.context["nars"]

    conclusion, truth = nars.infer(premises, rule)

    return {
        "conclusion": conclusion,
        "frequency": truth.frequency,
        "confidence": truth.confidence,
        "trace": nars.get_trace(),
    }


# =============================================================================
# OBJECT RESOLVERS
# =============================================================================

@observer_type.field("currentStyle")
async def resolve_observer_style(obj, info):
    """Resolve Observer's current style."""
    style_vector = obj.get("styleVector", [])

    if not style_vector or len(style_vector) < 33:
        style_vector = [0.33, 0.33, 0.34] + [0.11] * 9 + [0.2] * 5 + [0.0] * 16

    return vector_to_style(style_vector)


@observer_type.field("currentQualia")
async def resolve_observer_qualia(obj, info):
    """Resolve Observer's current qualia."""
    qualia_vector = obj.get("qualiaVector", [])

    if not qualia_vector or len(qualia_vector) < 17:
        qualia_vector = [0.5] * 17

    return vector_to_qualia(qualia_vector)


@observer_type.field("recentThoughts")
async def resolve_observer_recent_thoughts(obj, info, limit: int = 10):
    """Get Observer's recent thoughts."""
    kuzu: KuzuClient = info.context["kuzu"]

    return await kuzu.execute("""
        MATCH (o:Observer {id: 'ada'})-[:THINKS]->(t:Thought)
        RETURN t.id AS id, t.content AS content,
               t.timestamp AS timestamp, t.confidence AS confidence,
               t.step_number AS stepNumber, t.importance AS importance
        ORDER BY t.timestamp DESC
        LIMIT $limit
    """, {"limit": limit})


@observer_type.field("recentEpisodes")
async def resolve_observer_recent_episodes(obj, info, limit: int = 5):
    """Get Observer's recent episodes."""
    kuzu: KuzuClient = info.context["kuzu"]

    return await kuzu.execute("""
        MATCH (o:Observer {id: 'ada'})-[:REMEMBERS]->(e:Episode)
        RETURN e.id AS id, e.session_id AS sessionId,
               e.summary AS summary, e.start_time AS startTime,
               e.emotional_valence AS emotionalValence
        ORDER BY e.start_time DESC
        LIMIT $limit
    """, {"limit": limit})


@observer_type.field("reasoningTrace")
async def resolve_observer_trace(obj, info, depth: int = 10):
    """Get reasoning trace."""
    kuzu: KuzuClient = info.context["kuzu"]
    return await kuzu.get_reasoning_trace(depth)


@observer_type.field("totalThoughts")
async def resolve_observer_total_thoughts(obj, info):
    """Get total thought count."""
    kuzu: KuzuClient = info.context["kuzu"]

    result = await kuzu.execute("""
        MATCH (o:Observer {id: 'ada'})-[:THINKS]->(t:Thought)
        RETURN count(t) AS count
    """)

    return result[0]["count"] if result else 0


@observer_type.field("totalEpisodes")
async def resolve_observer_total_episodes(obj, info):
    """Get total episode count."""
    kuzu: KuzuClient = info.context["kuzu"]

    result = await kuzu.execute("""
        MATCH (o:Observer {id: 'ada'})-[:REMEMBERS]->(e:Episode)
        RETURN count(e) AS count
    """)

    return result[0]["count"] if result else 0


@thinking_style_type.field("dominantPearl")
def resolve_dominant_pearl(obj, info):
    """Get dominant PEARL mode."""
    pearl = obj.get("pearl", {})
    modes = {"SEE": pearl.get("see", 0), "DO": pearl.get("do", 0), "IMAGINE": pearl.get("imagine", 0)}
    return max(modes, key=modes.get)


@thinking_style_type.field("dominantRung")
def resolve_dominant_rung(obj, info):
    """Get dominant cognitive rung."""
    rung = obj.get("rung", {})
    values = [
        rung.get("r1_reflex", 0), rung.get("r2_affect", 0),
        rung.get("r3_pattern", 0), rung.get("r4_deliberate", 0),
        rung.get("r5_meta", 0), rung.get("r6_empathic", 0),
        rung.get("r7_counterfactual", 0), rung.get("r8_paradox", 0),
        rung.get("r9_transcend", 0),
    ]
    return values.index(max(values)) + 1 if values else 4


@thinking_style_type.field("dense")
def resolve_style_dense(obj, info):
    """Get dense vector representation."""
    return style_to_vector(obj)


@qualia_type.field("closestFamily")
def resolve_closest_family(obj, info):
    """Get closest qualia family."""
    # Simple heuristic based on warmth and arousal
    warmth = obj.get("warmth", 0.5)
    arousal = obj.get("arousal", 0.5)
    clarity = obj.get("clarity", 0.5)

    if warmth > 0.7:
        return "emberglow" if arousal > 0.5 else "woodwarm"
    elif clarity > 0.7:
        return "steelwind" if arousal > 0.5 else "frostbite"
    elif arousal > 0.7:
        return "sunburst" if warmth > 0.4 else "stormbreak"
    else:
        return "oceandrift"


@qualia_type.field("vector")
def resolve_qualia_vector(obj, info):
    """Get vector representation."""
    return qualia_to_vector(obj)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def vector_to_style(vec: List[float]) -> Dict:
    """Convert 33D vector to ThinkingStyle structure."""
    if len(vec) < 33:
        vec = vec + [0.0] * (33 - len(vec))

    return {
        "pearl": {"see": vec[0], "do": vec[1], "imagine": vec[2]},
        "rung": {
            "r1_reflex": vec[3], "r2_affect": vec[4], "r3_pattern": vec[5],
            "r4_deliberate": vec[6], "r5_meta": vec[7], "r6_empathic": vec[8],
            "r7_counterfactual": vec[9], "r8_paradox": vec[10], "r9_transcend": vec[11],
        },
        "sigma": {
            "observe": vec[12], "insight": vec[13], "belief": vec[14],
            "integrate": vec[15], "trajectory": vec[16],
        },
        "operations": {
            "abduct": vec[17], "deduce": vec[18], "synthesize": vec[19],
            "preflight": vec[20], "model_other": vec[21], "escalate": vec[22],
            "transcend": vec[23], "compress": vec[24],
        },
        "presence": {
            "authentic": vec[25], "performance": vec[26],
            "protective": vec[27], "integrated": vec[28],
        },
        "meta": {
            "confidence": vec[29], "exploration": vec[30],
            "novelty": vec[31], "counterfactual": vec[32],
        },
    }


def style_to_vector(style: Dict) -> List[float]:
    """Convert ThinkingStyle structure to 33D vector."""
    vec = []

    pearl = style.get("pearl", {})
    vec.extend([pearl.get("see", 0.33), pearl.get("do", 0.33), pearl.get("imagine", 0.34)])

    rung = style.get("rung", {})
    vec.extend([
        rung.get("r1_reflex", 0.11), rung.get("r2_affect", 0.11),
        rung.get("r3_pattern", 0.11), rung.get("r4_deliberate", 0.11),
        rung.get("r5_meta", 0.11), rung.get("r6_empathic", 0.11),
        rung.get("r7_counterfactual", 0.11), rung.get("r8_paradox", 0.11),
        rung.get("r9_transcend", 0.11),
    ])

    sigma = style.get("sigma", {})
    vec.extend([
        sigma.get("observe", 0.2), sigma.get("insight", 0.2),
        sigma.get("belief", 0.2), sigma.get("integrate", 0.2),
        sigma.get("trajectory", 0.2),
    ])

    ops = style.get("operations", {})
    vec.extend([
        ops.get("abduct", 0), ops.get("deduce", 0), ops.get("synthesize", 0),
        ops.get("preflight", 0), ops.get("model_other", 0), ops.get("escalate", 0),
        ops.get("transcend", 0), ops.get("compress", 0),
    ])

    pres = style.get("presence", {})
    vec.extend([
        pres.get("authentic", 0), pres.get("performance", 0),
        pres.get("protective", 0), pres.get("integrated", 0),
    ])

    meta = style.get("meta", {})
    vec.extend([
        meta.get("confidence", 0), meta.get("exploration", 0),
        meta.get("novelty", 0), meta.get("counterfactual", 0),
    ])

    return vec


def vector_to_qualia(vec: List[float]) -> Dict:
    """Convert 17D vector to Qualia structure."""
    if len(vec) < 17:
        vec = vec + [0.5] * (17 - len(vec))

    return {
        "arousal": vec[0], "valence": vec[1], "tension": vec[2],
        "warmth": vec[3], "clarity": vec[4], "boundary": vec[5],
        "depth": vec[6], "velocity": vec[7], "entropy": vec[8],
        "coherence": vec[9], "intimacy": vec[10], "presence": vec[11],
        "assertion": vec[12], "receptivity": vec[13], "groundedness": vec[14],
        "expansion": vec[15], "integration": vec[16],
        "family": None, "variant": None,
    }


def qualia_to_vector(qualia: Dict) -> List[float]:
    """Convert Qualia structure to 17D vector."""
    return [
        qualia.get("arousal", 0.5), qualia.get("valence", 0.5),
        qualia.get("tension", 0.5), qualia.get("warmth", 0.5),
        qualia.get("clarity", 0.5), qualia.get("boundary", 0.5),
        qualia.get("depth", 0.5), qualia.get("velocity", 0.5),
        qualia.get("entropy", 0.5), qualia.get("coherence", 0.5),
        qualia.get("intimacy", 0.5), qualia.get("presence", 0.5),
        qualia.get("assertion", 0.5), qualia.get("receptivity", 0.5),
        qualia.get("groundedness", 0.5), qualia.get("expansion", 0.5),
        qualia.get("integration", 0.5),
    ]


# =============================================================================
# SCHEMA FACTORY
# =============================================================================

def create_schema(kuzu: KuzuClient, lance: LanceClient, vsa: HypervectorSpace = None, nars: NARSReasoner = None):
    """
    Create executable GraphQL schema.

    Args:
        kuzu: Kuzu database client
        lance: LanceDB client
        vsa: VSA space (optional)
        nars: NARS reasoner (optional)

    Returns:
        Executable GraphQL schema
    """
    schema_path = Path(__file__).parent / "schema.graphql"
    type_defs = schema_path.read_text()

    return make_executable_schema(
        type_defs,
        query,
        mutation,
        observer_type,
        thought_type,
        episode_type,
        thinking_style_type,
        qualia_type,
        datetime_scalar,
        json_scalar,
    )
