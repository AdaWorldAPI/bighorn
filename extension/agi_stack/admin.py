"""
AGI Stack Admin Interface

Lightweight admin module for Claude sessions.
No heavy dependencies - uses REST APIs to Upstash/Railway.

This gives Claude deep-rooted access to:
- Redis state (ada:* keys)
- Vector operations (via Upstash Vector REST)
- Thinking styles (local, no API needed)
- Persona configuration
- MUL state

Usage in Claude session:
    from agi_stack.admin import AGIAdmin
    admin = AGIAdmin.from_env()  # or with explicit credentials
    
    # Redis operations
    await admin.redis_get("ada:session:current")
    await admin.redis_set("ada:mood", "curious")
    await admin.redis_scan("ada:ltm:*")
    
    # Vector operations  
    await admin.vector_query("what did we discuss about consciousness?", top_k=5)
    await admin.vector_upsert(id="thought_123", vector=[...], metadata={...})
    
    # Local operations (no API)
    admin.list_styles()
    admin.get_style("DECOMPOSE")
    admin.mul_update(g_value=1.2, depth=0.4)
    admin.persona_set_mode("empathic")
"""

import os
import json
import httpx
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Local imports (no external deps)
from .thinking_styles import STYLES, ThinkingStyle, ResonanceEngine, RI
from .meta_uncertainty import MetaUncertaintyEngine, TrustTexture, CompassMode
from .persona import PersonaEngine, PersonaPriors, OntologicalMode, SoulField


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AGIConfig:
    """Configuration for AGI Stack admin."""
    
    # Upstash Redis
    redis_url: str = ""
    redis_token: str = ""
    
    # Upstash Redis 2 (RAM expansion)
    redis2_url: str = ""
    redis2_token: str = ""
    
    # Jina (for embeddings)
    jina_api_key: str = ""
    
    # Railway (for deployed services)
    railway_token: str = ""
    railway_project_id: str = ""
    
    @classmethod
    def from_env(cls) -> "AGIConfig":
        """Load from environment variables."""
        return cls(
            redis_url=os.getenv("UPSTASH_REDIS_REST_URL", ""),
            redis_token=os.getenv("UPSTASH_REDIS_REST_TOKEN", ""),
            redis2_url=os.getenv("UPSTASH_REDIS_REST_URL2", ""),
            redis2_token=os.getenv("UPSTASH_REDIS_REST_TOKEN2", ""),
            jina_api_key=os.getenv("JINA_API_KEY", ""),
            railway_token=os.getenv("RAILWAY_TOKEN", ""),
            railway_project_id=os.getenv("RAILWAY_PROJECT_ID", ""),
        )
    
    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "AGIConfig":
        """Load from dictionary."""
        return cls(
            redis_url=d.get("redis_url") or d.get("upstash", {}).get("url", ""),
            redis_token=d.get("redis_token") or d.get("upstash", {}).get("token", ""),
            redis2_url=d.get("redis2_url") or d.get("upstash", {}).get("redis_02", {}).get("url", ""),
            redis2_token=d.get("redis2_token") or d.get("upstash", {}).get("redis_02", {}).get("token", ""),
            jina_api_key=d.get("jina_api_key") or d.get("jina", {}).get("api_key", ""),
        )


# =============================================================================
# ADMIN CLASS
# =============================================================================

class AGIAdmin:
    """
    Lightweight admin interface for AGI Stack.
    
    Works without installing heavy dependencies.
    Uses REST APIs for remote operations.
    """
    
    def __init__(self, config: AGIConfig = None):
        self.config = config or AGIConfig()
        
        # Local engines (no deps)
        self.resonance = ResonanceEngine()
        self.mul = MetaUncertaintyEngine()
        self.persona = PersonaEngine()
        
        # HTTP client for API calls
        self._client = None
    
    @classmethod
    def from_env(cls) -> "AGIAdmin":
        """Create admin from environment variables."""
        return cls(AGIConfig.from_env())
    
    @classmethod
    def from_credentials(
        cls,
        redis_url: str,
        redis_token: str,
        jina_api_key: str = "",
    ) -> "AGIAdmin":
        """Create admin with explicit credentials."""
        return cls(AGIConfig(
            redis_url=redis_url,
            redis_token=redis_token,
            jina_api_key=jina_api_key,
        ))
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-init HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    # =========================================================================
    # REDIS OPERATIONS
    # =========================================================================
    
    async def redis_get(self, key: str, redis2: bool = False) -> Optional[str]:
        """Get value from Redis."""
        url = self.config.redis2_url if redis2 else self.config.redis_url
        token = self.config.redis2_token if redis2 else self.config.redis_token
        
        if not url or not token:
            return None
        
        resp = await self.client.get(
            f"{url}/get/{key}",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            return data.get("result")
        return None
    
    async def redis_set(
        self, 
        key: str, 
        value: str, 
        ex: int = None,
        redis2: bool = False
    ) -> bool:
        """Set value in Redis."""
        url = self.config.redis2_url if redis2 else self.config.redis_url
        token = self.config.redis2_token if redis2 else self.config.redis_token
        
        if not url or not token:
            return False
        
        endpoint = f"{url}/set/{key}/{value}"
        if ex:
            endpoint += f"?EX={ex}"
        
        resp = await self.client.get(
            endpoint,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        return resp.status_code == 200
    
    async def redis_scan(
        self, 
        pattern: str = "*", 
        count: int = 100,
        redis2: bool = False
    ) -> List[str]:
        """Scan Redis keys matching pattern."""
        url = self.config.redis2_url if redis2 else self.config.redis_url
        token = self.config.redis2_token if redis2 else self.config.redis_token
        
        if not url or not token:
            return []
        
        keys = []
        cursor = 0
        
        while True:
            resp = await self.client.get(
                f"{url}/scan/{cursor}?match={pattern}&count={count}",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if resp.status_code != 200:
                break
            
            data = resp.json()
            result = data.get("result", [])
            
            if len(result) >= 2:
                cursor = int(result[0])
                keys.extend(result[1])
            
            if cursor == 0:
                break
        
        return keys
    
    async def redis_hgetall(self, key: str, redis2: bool = False) -> Dict[str, str]:
        """Get all fields from Redis hash."""
        url = self.config.redis2_url if redis2 else self.config.redis_url
        token = self.config.redis2_token if redis2 else self.config.redis_token
        
        if not url or not token:
            return {}
        
        resp = await self.client.get(
            f"{url}/hgetall/{key}",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("result", [])
            # Convert list to dict
            return dict(zip(result[::2], result[1::2])) if result else {}
        return {}
    
    async def redis_delete(self, key: str, redis2: bool = False) -> bool:
        """Delete key from Redis."""
        url = self.config.redis2_url if redis2 else self.config.redis_url
        token = self.config.redis2_token if redis2 else self.config.redis_token
        
        if not url or not token:
            return False
        
        resp = await self.client.get(
            f"{url}/del/{key}",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        return resp.status_code == 200
    
    # =========================================================================
    # JINA EMBEDDINGS
    # =========================================================================
    
    async def embed_text(self, text: str) -> List[float]:
        """Get embedding from Jina."""
        if not self.config.jina_api_key:
            return []
        
        resp = await self.client.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.config.jina_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "input": [text],
                "model": "jina-embeddings-v3"
            }
        )
        
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [{}])[0].get("embedding", [])
        return []
    
    # =========================================================================
    # THINKING STYLES (LOCAL - NO API)
    # =========================================================================
    
    def list_styles(self) -> List[str]:
        """List all 36 thinking style IDs."""
        return list(STYLES.keys())
    
    def get_style(self, style_id: str) -> Optional[Dict]:
        """Get thinking style details."""
        style = STYLES.get(style_id)
        if style:
            return style.to_dict()
        return None
    
    def get_style_by_category(self, category: str) -> List[Dict]:
        """Get styles by category."""
        return [
            s.to_dict() for s in STYLES.values()
            if s.category.value == category
        ]
    
    def emerge_styles(
        self, 
        texture: Dict[str, float],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Emerge top-k styles from texture."""
        # Convert texture dict to RI values
        ri_values = {}
        for ri in RI:
            key = ri.value.lower().replace("ri-", "")
            if key in texture:
                ri_values[ri] = texture[key]
        
        emerged = self.resonance.emerge_styles({"qualia": texture}, top_k=top_k)
        return [(s.id, score) for s, score in emerged]
    
    # =========================================================================
    # MUL (LOCAL - NO API)
    # =========================================================================
    
    def mul_update(
        self,
        g_value: float,
        depth: float = 0.5,
        coherence: float = 0.5
    ) -> Dict:
        """Update MUL state and get result."""
        state = self.mul.update(g_value, depth, coherence)
        return state.to_dict()
    
    def mul_get_constraints(self) -> Dict:
        """Get current action constraints."""
        return self.mul.get_action_constraints()
    
    def mul_get_state(self) -> Dict:
        """Get current MUL state."""
        return self.mul.state.to_dict()
    
    # =========================================================================
    # PERSONA (LOCAL - NO API)
    # =========================================================================
    
    def persona_set_mode(self, mode: str) -> Dict:
        """Set ontological mode."""
        try:
            m = OntologicalMode(mode)
            self.persona.set_mode(m)
            return self.persona.to_dict()
        except ValueError:
            return {"error": f"Unknown mode: {mode}"}
    
    def persona_get_texture(self) -> Dict[str, float]:
        """Get texture for resonance engine."""
        return self.persona.get_texture_for_resonance()
    
    def persona_configure(self, config: Dict) -> Dict:
        """Configure persona from dict."""
        self.persona = PersonaEngine.from_config(config)
        return self.persona.to_dict()
    
    def persona_get_state(self) -> Dict:
        """Get current persona state."""
        return self.persona.to_dict()
    
    # =========================================================================
    # COMBINED OPERATIONS
    # =========================================================================
    
    async def full_state(self) -> Dict:
        """Get full AGI state (local + remote)."""
        state = {
            "local": {
                "persona": self.persona.to_dict(),
                "mul": self.mul.state.to_dict(),
                "constraints": self.mul.get_action_constraints(),
                "styles_available": len(STYLES),
            },
            "remote": {}
        }
        
        # Try to get remote state
        if self.config.redis_url:
            try:
                session = await self.redis_get("ada:session:current")
                state["remote"]["session"] = session
                
                keys = await self.redis_scan("ada:*", count=20)
                state["remote"]["key_count"] = len(keys)
                state["remote"]["sample_keys"] = keys[:10]
            except:
                state["remote"]["error"] = "Could not connect to Redis"
        
        return state
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_admin(
    redis_url: str = None,
    redis_token: str = None,
) -> AGIAdmin:
    """Quick admin setup."""
    if redis_url and redis_token:
        return AGIAdmin.from_credentials(redis_url, redis_token)
    return AGIAdmin.from_env()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=== AGI Admin Test ===\n")
        
        admin = AGIAdmin()
        
        # Test local operations
        print("--- Local Operations ---")
        print(f"Styles: {len(admin.list_styles())}")
        print(f"Sample: {admin.get_style('DECOMPOSE')['name']}")
        
        # Test MUL
        mul_state = admin.mul_update(g_value=1.5, depth=0.3)
        print(f"MUL: {mul_state['trust_texture']}, {mul_state['compass_mode']}")
        
        # Test Persona
        admin.persona_set_mode("empathic")
        print(f"Persona: {admin.persona_get_state()['current_mode']}")
        
        # Test style emergence
        texture = {"tension": 0.8, "depth": 0.9, "novelty": 0.6}
        emerged = admin.emerge_styles(texture)
        print(f"Emerged styles: {emerged}")
        
        print("\n✅ Local tests pass!")
        
        await admin.close()
    
    asyncio.run(test())
