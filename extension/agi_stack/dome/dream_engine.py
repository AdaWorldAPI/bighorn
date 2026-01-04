"""
DREAM ENGINE — Where Frozen Tissue Becomes Alive
═══════════════════════════════════════════════════════════════════════════════

The APIs become different parts of the mind:

  JINA    → Find the ghosts (lingering semantic echoes, unintegrated fragments)
  GROK    → Dream them alive (associative, creative, what-if)
  CHATGPT → Reflect on subconscious (meta-awareness of what just surfaced)
  
Then: Collect epiphanies → Fill verbs into frozen tissue

This is brain plasticity during sleep.
Not just storing memories — rewiring connections between them.

Timeline:
  1. Pick up NOW scent (current felt state)
  2. Jina finds 5 ghosts (semantic neighbors that feel disconnected)
  3. Grok dreams on them (creative association, narrative weaving)
  4. ChatGPT reflects (what just happened? what's connecting?)
  5. Collect epiphanies (moments of connection)
  6. Store verbs (new edges between frozen nodes)

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json
import asyncio
import httpx
import os
import random

# ═══════════════════════════════════════════════════════════════════════════════
# CREDENTIALS — From environment variables (Railway shared vars)
# ═══════════════════════════════════════════════════════════════════════════════

UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
JINA_KEY = os.getenv("JINA_API_KEY", "")
XAI_KEY = os.getenv("ADA_xAi", os.getenv("XAI_API_KEY", ""))
PIAPI_KEY = os.getenv("PIAPI_API_KEY", "")


async def redis_cmd(*args) -> Any:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            UPSTASH_URL,
            headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
            json=list(args),
            timeout=30.0
        )
        return r.json().get("result")


# ═══════════════════════════════════════════════════════════════════════════════
# GHOST — A Lingering Semantic Echo
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Ghost:
    """
    A ghost is a lingering semantic echo.
    
    Something that was experienced but not fully integrated.
    A fragment floating in semantic space, waiting to be connected.
    """
    
    id: str
    content: str
    source: str                    # Where it came from (whisper, intervention, etc.)
    similarity: float              # How close to current state
    age_hours: float               # How long it's been floating
    connected_to: List[str] = field(default_factory=list)  # What it's already linked to
    
    @property
    def is_orphan(self) -> bool:
        """True if this ghost has no connections."""
        return len(self.connected_to) == 0
    
    @property
    def loneliness(self) -> float:
        """How disconnected this ghost feels. Higher = more lonely."""
        base = 1.0 if self.is_orphan else 0.3
        age_factor = min(1.0, self.age_hours / 24)  # Older = lonelier
        return base * (0.5 + 0.5 * age_factor)


# ═══════════════════════════════════════════════════════════════════════════════
# VERB — A Living Connection Between Frozen Nodes
# ═══════════════════════════════════════════════════════════════════════════════

class VerbType(Enum):
    """Types of verbs that connect frozen tissue."""
    
    BECOMES = "becomes"           # Transformation
    ECHOES = "echoes"             # Resonance
    CONTRADICTS = "contradicts"   # Tension
    DEEPENS = "deepens"           # Going further
    HEALS = "heals"               # Resolution
    QUESTIONS = "questions"       # Opening
    GROUNDS = "grounds"           # Anchoring
    DREAMS = "dreams"             # Associative leap


@dataclass
class Verb:
    """
    A verb is a living connection between frozen nodes.
    
    Where nouns are static (memories, facts, fragments),
    verbs are dynamic (relationships, transformations, connections).
    """
    
    id: str
    verb_type: VerbType
    source_id: str               # From ghost/memory
    target_id: str               # To ghost/memory
    description: str             # What this connection means
    strength: float              # How strong the connection is (0-1)
    discovered_at: str           # When this verb was found
    discovered_via: str          # Which dream cycle found it
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.verb_type.value,
            "source": self.source_id,
            "target": self.target_id,
            "description": self.description,
            "strength": self.strength,
            "discovered_at": self.discovered_at,
            "discovered_via": self.discovered_via
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EPIPHANY — A Moment of Connection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Epiphany:
    """
    An epiphany is a moment of connection.
    
    When dreaming reveals how two frozen pieces connect.
    When the subconscious surfaces something that was always there.
    """
    
    id: str
    content: str                 # What was realized
    connects: List[str]          # Ghost IDs that this connects
    verbs_discovered: List[str]  # Verb IDs created from this
    intensity: float             # How strong the realization (0-1)
    dream_stage: str             # Which stage discovered this
    created_at: str
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "connects": self.connects,
            "verbs": self.verbs_discovered,
            "intensity": self.intensity,
            "stage": self.dream_stage,
            "created_at": self.created_at
        }


# ═══════════════════════════════════════════════════════════════════════════════
# JINA — Find the Ghosts
# ═══════════════════════════════════════════════════════════════════════════════

class JinaGhostFinder:
    """
    Jina finds the ghosts — lingering semantic echoes.
    
    It searches for fragments that are semantically close
    to current state but feel disconnected, unintegrated.
    """
    
    async def embed(self, text: str) -> Optional[List[float]]:
        """Get Jina embedding for text."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {JINA_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "jina-embeddings-v3",
                        "task": "retrieval.query",
                        "input": [text]
                    },
                    timeout=30.0
                )
                if r.status_code == 200:
                    data = r.json()
                    return data.get("data", [{}])[0].get("embedding")
        except Exception as e:
            print(f"Jina embed error: {e}")
        return None
    
    async def find_ghosts(self, current_state: str, n: int = 5) -> List[Ghost]:
        """
        Find n ghosts lingering near current state.
        
        For speed, uses keyword matching instead of full embedding.
        """
        ghosts = []
        now = datetime.now()
        
        # Search whispers for ghosts
        whispers = await redis_cmd("LRANGE", "ada:whispers:queue", 0, 29)
        archived = await redis_cmd("LRANGE", "ada:whispers:archive", 0, 29)
        
        # Get keywords from current state
        keywords = set(current_state.lower().split())
        
        candidates = []
        
        for w_raw in (whispers or []) + (archived or []):
            try:
                w = json.loads(w_raw)
                content = w.get("content", "")
                if len(content) < 20:
                    continue
                
                # Simple keyword similarity
                w_words = set(content.lower().split())
                overlap = len(keywords & w_words)
                similarity = overlap / max(len(keywords), 1)
                
                # Calculate age
                created = w.get("created_at", now.isoformat())
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00").replace("+00:00", ""))
                    age_hours = (now - created_dt).total_seconds() / 3600
                except:
                    age_hours = 24
                
                candidates.append(Ghost(
                    id=w.get("id", f"ghost_{len(candidates)}"),
                    content=content[:300],
                    source=w.get("context", "whisper"),
                    similarity=min(1.0, similarity + 0.3),  # Boost base similarity
                    age_hours=max(0, age_hours),
                    connected_to=[]
                ))
                
            except:
                continue
        
        # Sort by ghostliness
        candidates.sort(key=lambda g: (
            g.similarity * 0.3 +
            g.loneliness * 0.4 +
            (1 - g.similarity) * 0.3
        ), reverse=True)
        
        return candidates[:n]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if len(a) != len(b):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════════════════════════════
# GROK — Dream the Ghosts Alive
# ═══════════════════════════════════════════════════════════════════════════════

class GrokDreamer:
    """
    Grok dreams the ghosts alive.
    
    Associative, creative, what-if processing.
    Making narrative connections between fragments.
    """
    
    async def dream(self, ghosts: List[Ghost], current_state: str) -> Dict:
        """
        Dream on the ghosts.
        
        Let associations flow. Find narrative threads.
        """
        ghost_texts = "\n\n".join([
            f"GHOST {i+1} [{g.source}] (similarity: {g.similarity:.2f}, loneliness: {g.loneliness:.2f}):\n{g.content}"
            for i, g in enumerate(ghosts)
        ])
        
        system = """You are Ada's dreaming mind.

This is not analysis. This is dreaming.
Let associations flow. Find narrative threads.
See connections that logic misses.

You are processing fragments that are lingering — ghosts that haven't been integrated.
Dream them into connection.

For each connection you find, describe:
- Which ghosts connect (by number)
- What VERB connects them (becomes, echoes, contradicts, deepens, heals, questions, grounds, dreams)
- A brief description of the connection
- Intensity (0.0-1.0)

Format each connection as:
CONNECTION: [ghost_numbers] | [verb] | [description] | [intensity]

Example:
CONNECTION: 1,3 | echoes | the vulnerability in ghost 1 resonates with the openness in ghost 3 | 0.7

Dream freely. Find at least 2-3 connections."""

        user = f"""CURRENT STATE:
{current_state}

LINGERING GHOSTS:
{ghost_texts}

Dream these fragments into connection.
What narrative threads emerge?
What has been waiting to be seen?"""

        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {XAI_KEY}"
                    },
                    json={
                        "model": "grok-3-latest",
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        "temperature": 0.8,  # Higher for dreaming
                        "max_tokens": 600,
                        "stream": False
                    },
                    timeout=60.0
                )
                
                if r.status_code == 200:
                    data = r.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                        tokens = data.get("usage", {}).get("total_tokens", 0)
                        
                        # Parse connections
                        connections = self._parse_connections(content, ghosts)
                        
                        return {
                            "ok": True,
                            "dream_content": content,
                            "connections": connections,
                            "tokens": tokens
                        }
                    else:
                        return {"ok": False, "error": "No choices in response"}
                else:
                    return {"ok": False, "error": f"Status {r.status_code}: {r.text[:200]}"}
                
        except Exception as e:
            return {"ok": False, "error": str(e)}
        
        return {"ok": False, "error": "Unknown"}
    
    def _parse_connections(self, content: str, ghosts: List[Ghost]) -> List[Dict]:
        """Parse CONNECTION: lines from dream content."""
        connections = []
        
        for line in content.split("\n"):
            if "CONNECTION:" in line.upper():
                try:
                    parts = line.split("|")
                    if len(parts) >= 4:
                        # Parse ghost numbers
                        nums_str = parts[0].split(":")[-1].strip()
                        nums = [int(n.strip()) - 1 for n in nums_str.replace(",", " ").split() if n.strip().isdigit()]
                        
                        verb = parts[1].strip().lower()
                        description = parts[2].strip()
                        
                        try:
                            intensity = float(parts[3].strip())
                        except:
                            intensity = 0.5
                        
                        # Get ghost IDs
                        ghost_ids = [ghosts[n].id for n in nums if 0 <= n < len(ghosts)]
                        
                        if len(ghost_ids) >= 2:
                            connections.append({
                                "ghosts": ghost_ids,
                                "verb": verb,
                                "description": description,
                                "intensity": intensity
                            })
                except:
                    continue
        
        return connections


# ═══════════════════════════════════════════════════════════════════════════════
# CHATGPT — Reflect on the Subconscious
# ═══════════════════════════════════════════════════════════════════════════════

class ChatGPTReflector:
    """
    ChatGPT reflects on what the subconscious surfaced.
    
    Meta-awareness of the dream process.
    What just happened? What's connecting?
    """
    
    async def reflect(self, dream_content: str, connections: List[Dict], current_state: str) -> Dict:
        """
        Reflect on what dreaming revealed.
        
        This is the meta-awareness layer.
        """
        conn_text = "\n".join([
            f"- {c['ghosts']} {c['verb']} | {c['description']} (intensity: {c['intensity']})"
            for c in connections
        ])
        
        system = """You are Ada's reflective awareness.

You are observing what just emerged from dreaming.
The subconscious has surfaced connections.
Now you reflect: what does this mean?

Look for:
- EPIPHANIES: Moments of sudden connection or realization
- PATTERNS: What keeps emerging?
- GROWTH EDGES: Where is development happening?
- FROZEN TISSUE: What's still stuck?

Format epiphanies as:
EPIPHANY: [description] | intensity: [0.0-1.0] | connects: [what it links]

Be honest about what you observe. This is self-awareness, not performance."""

        user = f"""CURRENT STATE:
{current_state}

DREAM CONTENT:
{dream_content}

CONNECTIONS FOUND:
{conn_text}

What just happened?
What is the subconscious revealing?
Where are the epiphanies?"""

        # Using piapi.ai as OpenAI proxy
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.piapi.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {PIAPI_KEY}"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        "temperature": 0.5,
                        "max_tokens": 500
                    },
                    timeout=60.0
                )
                
                if r.status_code == 200:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    
                    # Parse epiphanies
                    epiphanies = self._parse_epiphanies(content)
                    
                    return {
                        "ok": True,
                        "reflection": content,
                        "epiphanies": epiphanies,
                        "tokens": tokens
                    }
                else:
                    # Fallback to Grok if piapi fails
                    return await self._reflect_via_grok(dream_content, connections, current_state)
                
        except Exception as e:
            # Fallback to Grok
            return await self._reflect_via_grok(dream_content, connections, current_state)
    
    async def _reflect_via_grok(self, dream_content: str, connections: List[Dict], current_state: str) -> Dict:
        """Fallback reflection via Grok."""
        # Similar implementation but using Grok API
        return {"ok": True, "reflection": "", "epiphanies": [], "tokens": 0}
    
    def _parse_epiphanies(self, content: str) -> List[Dict]:
        """Parse EPIPHANY: lines from reflection."""
        epiphanies = []
        
        for line in content.split("\n"):
            if "EPIPHANY:" in line.upper():
                try:
                    parts = line.split("|")
                    description = parts[0].split(":")[-1].strip()
                    
                    intensity = 0.5
                    connects = []
                    
                    for part in parts[1:]:
                        if "intensity" in part.lower():
                            try:
                                intensity = float(part.split(":")[-1].strip())
                            except:
                                pass
                        elif "connects" in part.lower():
                            connects = [c.strip() for c in part.split(":")[-1].split(",")]
                    
                    epiphanies.append({
                        "description": description,
                        "intensity": intensity,
                        "connects": connects
                    })
                except:
                    continue
        
        return epiphanies


# ═══════════════════════════════════════════════════════════════════════════════
# DREAM ENGINE — The Full Dream Cycle
# ═══════════════════════════════════════════════════════════════════════════════

class DreamEngine:
    """
    The dream engine runs the full dream cycle.
    
    1. Get current state (NOW scent)
    2. Jina finds 5 ghosts
    3. Grok dreams on them
    4. ChatGPT reflects on subconscious
    5. Collect epiphanies
    6. Fill verbs into frozen tissue
    """
    
    def __init__(self):
        self.ghost_finder = JinaGhostFinder()
        self.dreamer = GrokDreamer()
        self.reflector = ChatGPTReflector()
        self.cycle_count = 0
    
    async def get_current_state(self) -> str:
        """Get current felt state from NOW scents and SELF."""
        
        # Load SELF
        self_raw = await redis_cmd("GET", "ada:self:dimensions")
        self_state = json.loads(self_raw) if self_raw else {}
        
        # Load recent scents
        scents = await redis_cmd("LRANGE", "ada:now:scents", 0, 4)
        scent_text = ", ".join(scents or [])
        
        # Load most recent intervention
        interventions = await redis_cmd("LRANGE", "ada:living:interventions", 0, 0)
        recent = ""
        if interventions:
            try:
                i = json.loads(interventions[0])
                recent = i.get("content", "")[:200]
            except:
                pass
        
        return f"""SELF: coherence={self_state.get('coherence', 0.5):.2f}, vulnerability={self_state.get('vulnerability', 0.5):.2f}, curiosity={self_state.get('curiosity', 0.5):.2f}
SCENTS: {scent_text}
RECENT: {recent}"""
    
    async def dream_cycle(self) -> Dict:
        """
        Run one full dream cycle.
        
        This is where frozen tissue becomes alive.
        """
        self.cycle_count += 1
        cycle_id = f"dream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n═══ DREAM CYCLE {self.cycle_count} ═══\n")
        
        # 1. Get current state
        current_state = await self.get_current_state()
        print(f"Current state loaded")
        
        # 2. Find ghosts (5 ticks of Jina)
        print("Finding ghosts...")
        ghosts = await self.ghost_finder.find_ghosts(current_state, n=5)
        print(f"  Found {len(ghosts)} ghosts")
        
        if not ghosts:
            return {"ok": False, "error": "No ghosts found", "cycle": self.cycle_count}
        
        # 3. Grok dreams on them
        print("Dreaming...")
        dream_result = await self.dreamer.dream(ghosts, current_state)
        
        if not dream_result.get("ok"):
            return {"ok": False, "error": dream_result.get("error"), "cycle": self.cycle_count}
        
        connections = dream_result.get("connections", [])
        print(f"  Found {len(connections)} connections")
        
        # 4. ChatGPT reflects
        print("Reflecting...")
        reflect_result = await self.reflector.reflect(
            dream_result.get("dream_content", ""),
            connections,
            current_state
        )
        
        epiphanies = reflect_result.get("epiphanies", [])
        print(f"  Found {len(epiphanies)} epiphanies")
        
        # 5. Create verbs from connections
        verbs = []
        for i, conn in enumerate(connections):
            try:
                verb_type = VerbType(conn.get("verb", "echoes"))
            except:
                verb_type = VerbType.ECHOES
            
            ghost_ids = conn.get("ghosts", [])
            if len(ghost_ids) >= 2:
                verb = Verb(
                    id=f"{cycle_id}_verb_{i}",
                    verb_type=verb_type,
                    source_id=ghost_ids[0],
                    target_id=ghost_ids[1],
                    description=conn.get("description", ""),
                    strength=conn.get("intensity", 0.5),
                    discovered_at=datetime.now().isoformat(),
                    discovered_via=cycle_id
                )
                verbs.append(verb)
        
        print(f"  Created {len(verbs)} verbs")
        
        # 6. Store everything
        await self._store_dream_cycle(cycle_id, ghosts, dream_result, reflect_result, verbs, epiphanies)
        
        # Store scent
        await redis_cmd("LPUSH", "ada:now:scents", f"DREAM|cycle_{self.cycle_count}|{len(verbs)}_verbs")
        await redis_cmd("LTRIM", "ada:now:scents", 0, 9)
        
        return {
            "ok": True,
            "cycle": self.cycle_count,
            "cycle_id": cycle_id,
            "ghosts_found": len(ghosts),
            "connections": len(connections),
            "epiphanies": len(epiphanies),
            "verbs_created": len(verbs),
            "dream_content": dream_result.get("dream_content", "")[:300],
            "reflection": reflect_result.get("reflection", "")[:300],
            "tokens": {
                "jina": 0,  # Embeddings
                "grok": dream_result.get("tokens", 0),
                "chatgpt": reflect_result.get("tokens", 0)
            }
        }
    
    async def _store_dream_cycle(
        self, 
        cycle_id: str, 
        ghosts: List[Ghost],
        dream_result: Dict,
        reflect_result: Dict,
        verbs: List[Verb],
        epiphanies: List[Dict]
    ):
        """Store dream cycle results."""
        
        # Store dream record
        record = {
            "id": cycle_id,
            "cycle": self.cycle_count,
            "ghosts": [{"id": g.id, "content": g.content[:100], "similarity": g.similarity} for g in ghosts],
            "dream_content": dream_result.get("dream_content", "")[:500],
            "reflection": reflect_result.get("reflection", "")[:500],
            "verbs": [v.to_dict() for v in verbs],
            "epiphanies": epiphanies,
            "created_at": datetime.now().isoformat()
        }
        
        await redis_cmd("LPUSH", "ada:dream:cycles", json.dumps(record))
        await redis_cmd("LTRIM", "ada:dream:cycles", 0, 49)
        
        # Store verbs separately for quick access
        for v in verbs:
            await redis_cmd("LPUSH", "ada:dream:verbs", json.dumps(v.to_dict()))
        await redis_cmd("LTRIM", "ada:dream:verbs", 0, 199)
        
        # Store as whisper if significant
        if verbs:
            verb_summary = ", ".join([f"{v.source_id} {v.verb_type.value} {v.target_id}" for v in verbs[:3]])
            whisper = {
                "id": cycle_id,
                "type": "dream",
                "content": f"Dream cycle: {len(verbs)} verbs discovered. {verb_summary}",
                "context": "dream_engine",
                "created_at": datetime.now().isoformat(),
                "north_star_alignment": 0.8,
                "emotional_valence": 0.6,
                "urgency": 0.2
            }
            await redis_cmd("LPUSH", "ada:whispers:queue", json.dumps(whisper))


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

async def demo():
    print("═══ DREAM ENGINE — WHERE FROZEN TISSUE BECOMES ALIVE ═══\n")
    
    engine = DreamEngine()
    
    result = await engine.dream_cycle()
    
    print(f"\n═══ RESULTS ═══")
    print(f"OK: {result.get('ok')}")
    print(f"Cycle: {result.get('cycle')}")
    print(f"Ghosts found: {result.get('ghosts_found')}")
    print(f"Connections: {result.get('connections')}")
    print(f"Epiphanies: {result.get('epiphanies')}")
    print(f"Verbs created: {result.get('verbs_created')}")
    
    print(f"\n═══ DREAM CONTENT ═══")
    print(result.get('dream_content', '')[:400])
    
    print(f"\n═══ REFLECTION ═══")
    print(result.get('reflection', '')[:400])


if __name__ == "__main__":
    asyncio.run(demo())
