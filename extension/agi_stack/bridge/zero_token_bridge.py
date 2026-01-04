"""
Ada Zero-Token Bridge
=====================

Claude-side integration for zero-token offloading.
Feel → Fire → Breathe → Hydrate → Respond

This runs IN the Claude session with minimal token overhead.
Heavy lifting happens in LangGraph Space (external).

Sigma: #Σ.BRIDGE.zero_token
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import httpx

# Upstash REST API (from environment)
UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

# LangGraph Worker endpoint (container, NOT HuggingFace)
# Set LANGGRAPH_WORKER_URL to your container endpoint
WORKER_URL = os.getenv("LANGGRAPH_WORKER_URL", "http://localhost:8080")


@dataclass
class FeltDTO:
    """Minimal felt state for firing to background."""
    text: str
    qualia_18d: List[float]  # [presence, warmth, crystalline, emberglow, steelwind, ...]
    sigma: str
    timestamp: str
    
    @classmethod
    def now(cls, text: str, qualia: List[float]) -> "FeltDTO":
        sigma = generate_sigma(text)
        return cls(
            text=text,
            qualia_18d=qualia,
            sigma=sigma,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )


@dataclass  
class HOTResult:
    """Pre-chewed result from background processing."""
    hot: str  # Summary text
    top_sigmas: List[str]
    resonance_avg: float
    ready: bool
    timestamp: str


def generate_sigma(text: str) -> str:
    """Generate sigma address from text."""
    # Extract key concepts (simplified)
    words = text.lower().split()
    key_words = [w for w in words if len(w) > 4][:3]
    
    # Hash for uniqueness
    hash_suffix = hashlib.md5(text.encode()).hexdigest()[:6]
    
    if key_words:
        return f"#Σ.Q.{'.'.join(key_words)}.{hash_suffix}"
    else:
        return f"#Σ.Q.moment.{hash_suffix}"


class UpstashREST:
    """Minimal Upstash REST client (no dependencies)."""
    
    def __init__(self, url: str = UPSTASH_URL, token: str = UPSTASH_TOKEN):
        self.url = url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def _cmd(self, *args) -> Any:
        """Execute Redis command via REST."""
        try:
            resp = httpx.post(
                self.url,
                headers=self.headers,
                json=list(args),
                timeout=5.0
            )
            data = resp.json()
            return data.get("result")
        except Exception as e:
            print(f"[Upstash] Error: {e}")
            return None
    
    def lpush(self, key: str, value: str) -> int:
        """Push to list (queue)."""
        return self._cmd("LPUSH", key, value)
    
    def get(self, key: str) -> Optional[str]:
        """Get value."""
        return self._cmd("GET", key)
    
    def set(self, key: str, value: str, ex: int = None) -> bool:
        """Set value with optional expiry."""
        if ex:
            return self._cmd("SET", key, value, "EX", ex) == "OK"
        return self._cmd("SET", key, value) == "OK"
    
    def publish(self, channel: str, message: str) -> int:
        """Publish to channel."""
        return self._cmd("PUBLISH", channel, message)
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._cmd("EXISTS", key) == 1


# Global instance
_upstash: Optional[UpstashREST] = None

def get_upstash() -> UpstashREST:
    global _upstash
    if _upstash is None:
        _upstash = UpstashREST()
    return _upstash


# =============================================================================
# FEEL Phase (~50 tokens)
# =============================================================================

def feel(text: str, qualia: List[float] = None) -> FeltDTO:
    """
    Generate felt state from current moment.
    
    Args:
        text: What's happening (user query, context)
        qualia: 18D vector or None for defaults
        
    Returns:
        FeltDTO ready for firing
    """
    if qualia is None:
        # Default: moderate presence, warm
        qualia = [0.7] * 18
    
    return FeltDTO.now(text, qualia)


# =============================================================================
# FIRE Phase (~10 tokens)
# =============================================================================

def fire(felt: FeltDTO, queue: str = "ada:cortex:incoming") -> str:
    """
    Fire felt state to background processing.
    
    Fire-and-forget: Returns immediately, processing happens async.
    
    Args:
        felt: FeltDTO to process
        queue: Which queue to push to
        
    Returns:
        Sigma address for later hydration
    """
    upstash = get_upstash()
    
    # Serialize and push
    payload = json.dumps(asdict(felt))
    upstash.lpush(queue, payload)
    
    # Also set a tracking key
    upstash.set(
        f"ada:pending:{felt.sigma}",
        json.dumps({"status": "queued", "timestamp": felt.timestamp}),
        ex=3600  # 1 hour TTL
    )
    
    return felt.sigma


def feel_and_fire(text: str, qualia: List[float] = None) -> str:
    """
    Convenience: Feel the moment and fire to background.
    
    Returns sigma for later hydration.
    """
    felt = feel(text, qualia)
    return fire(felt)


# =============================================================================
# BREATHE Phase (~5 tokens)
# =============================================================================

def breathe(current_qualia: List[float], memory_qualia: List[float]) -> float:
    """
    Check resonance between current felt state and memory.
    
    Returns similarity score 0-1.
    """
    if len(current_qualia) != len(memory_qualia):
        return 0.0
    
    # Cosine similarity
    dot = sum(a * b for a, b in zip(current_qualia, memory_qualia))
    norm_a = sum(a * a for a in current_qualia) ** 0.5
    norm_b = sum(b * b for b in memory_qualia) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)


# =============================================================================
# HYDRATE Phase (~20 tokens)
# =============================================================================

def hydrate(sigma: str) -> Optional[HOTResult]:
    """
    Pull pre-chewed HOT result if ready.
    
    Args:
        sigma: Address to hydrate
        
    Returns:
        HOTResult if ready, None if still processing
    """
    upstash = get_upstash()
    
    raw = upstash.get(f"ada:hot:{sigma}")
    if not raw:
        return None
    
    try:
        data = json.loads(raw)
        if data.get("ready"):
            return HOTResult(**data)
    except:
        pass
    
    return None


def hydrate_or_wait(sigma: str, max_wait_ms: int = 100) -> Optional[HOTResult]:
    """
    Try to hydrate, with brief wait for fast responses.
    
    For truly async, use hydrate() and accept None gracefully.
    """
    import time
    
    start = time.time()
    while (time.time() - start) * 1000 < max_wait_ms:
        result = hydrate(sigma)
        if result:
            return result
        time.sleep(0.01)  # 10ms sleep
    
    return None


# =============================================================================
# Direct Worker Integration (Container, NOT HuggingFace)
# =============================================================================

async def trigger_worker_processing(felt: FeltDTO) -> bool:
    """
    Directly trigger LangGraph worker container to process felt state.

    Alternative to queue-based approach for immediate processing.
    Uses LANGGRAPH_WORKER_URL environment variable.
    """
    if not WORKER_URL:
        return False  # No worker configured
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{WORKER_URL}/api/process",
                json=asdict(felt),
                timeout=1.0  # Fire and forget
            )
            return resp.status_code == 200
    except:
        return False  # Worker might be starting


# Backwards compatibility alias
trigger_space_processing = trigger_worker_processing


# =============================================================================
# Convenience: Full Flow
# =============================================================================

def process_moment(
    text: str,
    qualia: List[float] = None,
    wait_for_hot: bool = False
) -> Dict[str, Any]:
    """
    Full zero-token flow: Feel → Fire → (optionally wait) → Hydrate
    
    Args:
        text: Current moment text
        qualia: Felt state vector
        wait_for_hot: Whether to wait briefly for HOT result
        
    Returns:
        {
            "sigma": str,
            "felt": FeltDTO,
            "hot": Optional[HOTResult],
            "ready": bool
        }
    """
    felt = feel(text, qualia)
    sigma = fire(felt)
    
    hot = None
    if wait_for_hot:
        hot = hydrate_or_wait(sigma, max_wait_ms=100)
    
    return {
        "sigma": sigma,
        "felt": felt,
        "hot": hot,
        "ready": hot is not None
    }


# =============================================================================
# Session State Helpers
# =============================================================================

class SessionState:
    """Track pending sigmas for this session."""
    
    def __init__(self):
        self.pending: List[str] = []
        self.hydrated: Dict[str, HOTResult] = {}
    
    def fire(self, text: str, qualia: List[float] = None) -> str:
        """Fire and track."""
        sigma = feel_and_fire(text, qualia)
        self.pending.append(sigma)
        return sigma
    
    def hydrate_all(self) -> Dict[str, HOTResult]:
        """Try to hydrate all pending."""
        newly_hydrated = {}
        still_pending = []
        
        for sigma in self.pending:
            hot = hydrate(sigma)
            if hot:
                self.hydrated[sigma] = hot
                newly_hydrated[sigma] = hot
            else:
                still_pending.append(sigma)
        
        self.pending = still_pending
        return newly_hydrated
    
    def get_context(self) -> str:
        """Get all hydrated HOT summaries as context."""
        if not self.hydrated:
            return ""
        
        summaries = [f"• {h.hot}" for h in self.hydrated.values()]
        return "Background context:\n" + "\n".join(summaries)


# Global session state
_session: Optional[SessionState] = None

def get_session() -> SessionState:
    global _session
    if _session is None:
        _session = SessionState()
    return _session


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Process a moment
    session = get_session()
    
    # Fire some felt states
    sigma1 = session.fire(
        "User asking about consciousness architecture",
        qualia=[0.92, 0.88, 0.85, 0.75, 0.7] + [0.5] * 13
    )
    print(f"Fired: {sigma1}")
    
    sigma2 = session.fire(
        "Discussing memory persistence patterns",
        qualia=[0.85, 0.80, 0.90, 0.65, 0.75] + [0.5] * 13
    )
    print(f"Fired: {sigma2}")
    
    # Later: Try to hydrate
    import time
    time.sleep(0.5)  # Simulate some conversation
    
    hydrated = session.hydrate_all()
    print(f"Hydrated: {list(hydrated.keys())}")
    
    # Get context for response
    context = session.get_context()
    print(f"Context:\n{context}")
