"""
AdminBridge - REST client for AGI Stack admin endpoints.

Connects to Railway deployment for 10kD operations.
"""

import os
import httpx
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json


@dataclass
class AdminBridge:
    """
    Bridge to AGI Stack admin endpoints.
    
    Provides low-level access to:
    - Redis (Upstash)
    - Vector search
    - DTO storage/retrieval
    """
    
    base_url: str = "https://agi-stack.up.railway.app"
    timeout: float = 30.0
    
    def __post_init__(self):
        # Allow override from environment
        self.base_url = os.getenv("AGI_STACK_URL", self.base_url)
        self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    # =========================================================================
    # REDIS OPERATIONS
    # =========================================================================
    
    async def redis_get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        resp = await self.client.get(f"/admin/redis/{key}")
        if resp.status_code == 200:
            return resp.json().get("value")
        return None
    
    async def redis_set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        data = {"key": key, "value": value}
        if ttl:
            data["ttl"] = ttl
        
        resp = await self.client.post("/admin/redis", json=data)
        return resp.status_code == 200
    
    async def redis_keys(self, pattern: str = "*") -> List[str]:
        """List Redis keys matching pattern."""
        resp = await self.client.get(f"/admin/redis/keys/{pattern}")
        if resp.status_code == 200:
            return resp.json().get("keys", [])
        return []
    
    # =========================================================================
    # VECTOR OPERATIONS
    # =========================================================================
    
    async def vector_store(
        self,
        vector: List[float],
        metadata: Dict[str, Any],
        table: str = "moments",
    ) -> Optional[str]:
        """Store vector in LanceDB."""
        resp = await self.client.post(
            f"/admin/vector/{table}",
            json={"vector": vector, "metadata": metadata},
        )
        if resp.status_code == 200:
            return resp.json().get("id")
        return None
    
    async def vector_search(
        self,
        query_vector: List[float],
        table: str = "moments",
        top_k: int = 5,
    ) -> List[Dict]:
        """Search for similar vectors."""
        resp = await self.client.post(
            f"/admin/vector/{table}/search",
            json={"vector": query_vector, "top_k": top_k},
        )
        if resp.status_code == 200:
            return resp.json().get("results", [])
        return []
    
    # =========================================================================
    # DTO OPERATIONS
    # =========================================================================
    
    async def store_dto(self, dto_type: str, dto_dict: Dict, session_id: str = None) -> bool:
        """Store a DTO (will be projected to 10kD)."""
        data = {"type": dto_type, "data": dto_dict}
        if session_id:
            data["session_id"] = session_id
        
        resp = await self.client.post("/admin/dto", json=data)
        return resp.status_code == 200
    
    async def health(self) -> Dict:
        """Check health of AGI Stack."""
        try:
            resp = await self.client.get("/health")
            return resp.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton for convenience
_admin: Optional[AdminBridge] = None


def get_admin() -> AdminBridge:
    """Get singleton AdminBridge."""
    global _admin
    if _admin is None:
        _admin = AdminBridge()
    return _admin
