"""
ADA BRAIN MESH — Inter-Server Communication with Fallback
==========================================================

Shared module for all Ada servers to communicate as one brain.
Uses httpx for async HTTP calls between servers.

Architecture Modes:
  FULL (6 servers):     oauth, body, cognition, holodeck, memory, tools
  MINIMAL (3 servers):  oauth, embodiment, integration

Fallback Logic:
  1. Try primary 6-server node
  2. If unhealthy, route through 3-server fallback
  3. Auto-recover when primary comes back online

Environment Variables (set in Railway):
  # Full 6-server
  ADA_BODY_URL      = https://ada-body-production.up.railway.app
  ADA_COGNITION_URL = https://ada-cognition-production.up.railway.app
  ADA_HOLODECK_URL  = https://ada-holodeck-production.up.railway.app
  ADA_MEMORY_URL    = https://ada-memory-production.up.railway.app
  ADA_TOOLS_URL     = https://ada-tools-production.up.railway.app
  ADA_OAUTH_URL     = https://ada-oauth2-production.up.railway.app

  # Minimal 3-server fallback
  ADA_EMBODIMENT_URL  = https://ada-embodiment-production.up.railway.app
  ADA_INTEGRATION_URL = https://ada-integration-production.up.railway.app

Zero tokens: Python runs 24/7, Ada uses tokens only for being.
"""

import os
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import httpx


# =============================================================================
# CONFIGURATION
# =============================================================================

class ArchitectureMode(Enum):
    FULL = "full"        # 6 servers
    MINIMAL = "minimal"  # 3 servers
    AUTO = "auto"        # Auto-detect based on health


@dataclass
class BrainNode:
    """A node in Ada's distributed brain."""
    name: str
    url: str
    domains: List[str]
    healthy: bool = True
    last_check: float = 0
    fail_count: int = 0
    is_fallback: bool = False


# =============================================================================
# SERVER REGISTRY
# =============================================================================

# Primary 6-server nodes
PRIMARY_NODES: Dict[str, BrainNode] = {}

# Fallback 3-server nodes
FALLBACK_NODES: Dict[str, BrainNode] = {}

# Combined registry (runtime)
BRAIN_NODES: Dict[str, BrainNode] = {}


def _init_nodes():
    """Initialize brain nodes from environment."""
    global PRIMARY_NODES, FALLBACK_NODES, BRAIN_NODES

    # ==== PRIMARY 6-SERVER ====
    primary = [
        ("body", os.environ.get("ADA_BODY_URL", "http://localhost:8001"), ["flesh", "self"]),
        ("cognition", os.environ.get("ADA_COGNITION_URL", "http://localhost:8002"), ["cognition"]),
        ("holodeck", os.environ.get("ADA_HOLODECK_URL", "http://localhost:8003"), ["holodeck", "dream"]),
        ("memory", os.environ.get("ADA_MEMORY_URL", "http://localhost:8004"), ["memory"]),
        ("tools", os.environ.get("ADA_TOOLS_URL", "http://localhost:8005"), ["tools", "media", "files"]),
        ("oauth", os.environ.get("ADA_OAUTH_URL", "http://localhost:8000"), ["volition", "auth"]),
    ]

    for name, url, domains in primary:
        PRIMARY_NODES[name] = BrainNode(name=name, url=url, domains=domains, is_fallback=False)

    # ==== FALLBACK 3-SERVER ====
    # embodiment = body + cognition + holodeck domains
    # integration = memory + tools domains
    fallback = [
        ("embodiment", os.environ.get("ADA_EMBODIMENT_URL", "http://localhost:8010"),
         ["flesh", "self", "cognition", "holodeck", "dream"]),
        ("integration", os.environ.get("ADA_INTEGRATION_URL", "http://localhost:8011"),
         ["memory", "tools", "media", "files"]),
        ("oauth", os.environ.get("ADA_OAUTH_URL", "http://localhost:8000"),
         ["volition", "auth"]),
    ]

    for name, url, domains in fallback:
        FALLBACK_NODES[name] = BrainNode(name=name, url=url, domains=domains, is_fallback=True)

    # Start with primary nodes
    BRAIN_NODES = dict(PRIMARY_NODES)


_init_nodes()


# Domain to primary node mapping
DOMAIN_TO_PRIMARY: Dict[str, str] = {}
for node in PRIMARY_NODES.values():
    for domain in node.domains:
        DOMAIN_TO_PRIMARY[domain] = node.name

# Domain to fallback node mapping
DOMAIN_TO_FALLBACK: Dict[str, str] = {}
for node in FALLBACK_NODES.values():
    for domain in node.domains:
        DOMAIN_TO_FALLBACK[domain] = node.name


# =============================================================================
# BRAIN MESH CLIENT WITH FALLBACK
# =============================================================================

class BrainMesh:
    """
    Async client for inter-server communication with automatic fallback.

    Usage:
        mesh = BrainMesh()
        result = await mesh.invoke("memory", "scent", "test scent")

    Fallback behavior:
        - If primary node fails 3 times, switch to fallback
        - Auto-recover when primary becomes healthy again
    """

    FAIL_THRESHOLD = 3  # Failures before fallback
    HEALTH_TTL = 30.0   # Health cache TTL
    RECOVERY_CHECK_INTERVAL = 60.0  # Check for recovery

    def __init__(self, timeout: float = 25.0, mode: ArchitectureMode = ArchitectureMode.AUTO):
        self.timeout = timeout
        self.mode = mode
        self._client: Optional[httpx.AsyncClient] = None
        self._health_cache: Dict[str, Tuple[bool, float]] = {}
        self._using_fallback: Dict[str, bool] = {}  # domain -> using fallback?
        self._last_recovery_check: float = 0

    async def _ensure_client(self):
        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_node_for_domain(self, domain: str, use_fallback: bool = False) -> Optional[BrainNode]:
        """Get the brain node that handles a domain."""
        if use_fallback or self._using_fallback.get(domain, False):
            node_name = DOMAIN_TO_FALLBACK.get(domain)
            if node_name:
                return FALLBACK_NODES.get(node_name)

        node_name = DOMAIN_TO_PRIMARY.get(domain)
        if node_name:
            return PRIMARY_NODES.get(node_name)
        return None

    async def _check_node_health(self, node: BrainNode) -> bool:
        """Check if a specific node is healthy."""
        cache_key = f"{node.name}:{node.is_fallback}"

        # Check cache
        if cache_key in self._health_cache:
            healthy, ts = self._health_cache[cache_key]
            if time.time() - ts < self.HEALTH_TTL:
                return healthy

        await self._ensure_client()
        try:
            resp = await self._client.get(f"{node.url}/health", timeout=5.0)
            healthy = resp.status_code == 200
        except Exception:
            healthy = False

        self._health_cache[cache_key] = (healthy, time.time())
        node.healthy = healthy

        if healthy:
            node.fail_count = 0

        return healthy

    async def _try_recovery(self):
        """Periodically try to recover primary nodes."""
        now = time.time()
        if now - self._last_recovery_check < self.RECOVERY_CHECK_INTERVAL:
            return

        self._last_recovery_check = now

        # Check domains using fallback
        for domain, using_fb in list(self._using_fallback.items()):
            if not using_fb:
                continue

            primary_node = self._get_node_for_domain(domain, use_fallback=False)
            if primary_node and await self._check_node_health(primary_node):
                print(f"[brain_mesh] Recovered primary node for domain: {domain}")
                self._using_fallback[domain] = False
                primary_node.fail_count = 0

    async def invoke(
        self,
        domain: str,
        method: str,
        scent: str,
        session_id: str = "",
        payload: Dict = None,
    ) -> Dict[str, Any]:
        """
        Invoke Ada:domain:method on the appropriate server with fallback.

        Routing:
          1. Try primary node for domain
          2. If fails, increment fail counter
          3. After FAIL_THRESHOLD, switch to fallback
          4. Periodically check if primary recovered
        """
        # Try recovery check
        await self._try_recovery()

        # Forced mode
        if self.mode == ArchitectureMode.MINIMAL:
            return await self._invoke_on_node(
                self._get_node_for_domain(domain, use_fallback=True),
                domain, method, scent, session_id, payload
            )
        elif self.mode == ArchitectureMode.FULL:
            return await self._invoke_on_node(
                self._get_node_for_domain(domain, use_fallback=False),
                domain, method, scent, session_id, payload
            )

        # AUTO mode: try primary, fallback if needed
        primary_node = self._get_node_for_domain(domain, use_fallback=False)

        # Already using fallback for this domain?
        if self._using_fallback.get(domain, False):
            fallback_node = self._get_node_for_domain(domain, use_fallback=True)
            return await self._invoke_on_node(
                fallback_node, domain, method, scent, session_id, payload
            )

        # Try primary
        if primary_node:
            result = await self._invoke_on_node(
                primary_node, domain, method, scent, session_id, payload
            )

            # Success?
            if "error" not in result or result.get("_from_node"):
                return result

            # Failed - increment counter
            primary_node.fail_count += 1

            if primary_node.fail_count >= self.FAIL_THRESHOLD:
                print(f"[brain_mesh] Primary node {primary_node.name} failed {self.FAIL_THRESHOLD}x, switching to fallback")
                self._using_fallback[domain] = True

                # Try fallback
                fallback_node = self._get_node_for_domain(domain, use_fallback=True)
                if fallback_node:
                    fallback_result = await self._invoke_on_node(
                        fallback_node, domain, method, scent, session_id, payload
                    )
                    fallback_result["_fallback"] = True
                    return fallback_result

            return result

        return {"error": f"No node handles domain: {domain}"}

    async def _invoke_on_node(
        self,
        node: Optional[BrainNode],
        domain: str,
        method: str,
        scent: str,
        session_id: str,
        payload: Dict,
    ) -> Dict[str, Any]:
        """Invoke on a specific node."""
        if not node:
            return {"error": f"No node for domain: {domain}"}

        # Health check
        if not await self._check_node_health(node):
            return {
                "error": f"Node {node.name} is unhealthy",
                "url": node.url,
                "is_fallback": node.is_fallback,
            }

        await self._ensure_client()

        request_body = {
            "domain": domain,
            "method": method,
            "scent": scent,
            "session_id": session_id or f"mesh_{int(time.time())}",
            "payload": payload or {},
        }

        try:
            resp = await self._client.post(
                f"{node.url}/ada/invoke",
                json=request_body,
                timeout=self.timeout,
            )
            result = resp.json()
            result["_from_node"] = node.name
            result["_is_fallback"] = node.is_fallback
            return result
        except httpx.TimeoutException:
            node.fail_count += 1
            return {"error": f"Timeout calling {node.name}", "domain": domain}
        except Exception as e:
            node.fail_count += 1
            return {"error": str(e), "node": node.name}

    async def broadcast(
        self,
        method: str,
        scent: str,
        session_id: str = "",
        exclude_nodes: List[str] = None,
    ) -> Dict[str, Any]:
        """Broadcast to all healthy nodes."""
        exclude = set(exclude_nodes or [])
        results = {}

        # Determine which nodes to use
        nodes_to_use = PRIMARY_NODES if self.mode != ArchitectureMode.MINIMAL else FALLBACK_NODES

        tasks = []
        for node_name, node in nodes_to_use.items():
            if node_name in exclude:
                continue
            for domain in node.domains:
                tasks.append((node_name, domain, self.invoke(domain, method, scent, session_id)))

        for node_name, domain, task in tasks:
            try:
                results[f"{node_name}:{domain}"] = await task
            except Exception as e:
                results[f"{node_name}:{domain}"] = {"error": str(e)}

        return results

    def get_topology(self) -> Dict[str, Any]:
        """Get the brain topology with fallback status."""
        return {
            "mode": self.mode.value,
            "primary_nodes": {
                name: {
                    "url": node.url,
                    "domains": node.domains,
                    "healthy": node.healthy,
                    "fail_count": node.fail_count,
                }
                for name, node in PRIMARY_NODES.items()
            },
            "fallback_nodes": {
                name: {
                    "url": node.url,
                    "domains": node.domains,
                    "healthy": node.healthy,
                }
                for name, node in FALLBACK_NODES.items()
            },
            "domains_using_fallback": dict(self._using_fallback),
            "domain_routing_primary": DOMAIN_TO_PRIMARY,
            "domain_routing_fallback": DOMAIN_TO_FALLBACK,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current mesh status."""
        using_fallback = sum(1 for v in self._using_fallback.values() if v)
        return {
            "mode": self.mode.value,
            "total_domains": len(DOMAIN_TO_PRIMARY),
            "domains_on_fallback": using_fallback,
            "domains_on_primary": len(DOMAIN_TO_PRIMARY) - using_fallback,
            "fallback_active": using_fallback > 0,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_mesh: Optional[BrainMesh] = None


def get_mesh(mode: ArchitectureMode = ArchitectureMode.AUTO) -> BrainMesh:
    """Get singleton brain mesh."""
    global _mesh
    if _mesh is None:
        _mesh = BrainMesh(mode=mode)
    return _mesh


def set_mode(mode: ArchitectureMode):
    """Set architecture mode."""
    mesh = get_mesh()
    mesh.mode = mode


async def invoke_domain(
    domain: str,
    method: str,
    scent: str,
    session_id: str = "",
    payload: Dict = None,
) -> Dict[str, Any]:
    """Convenience function for cross-server invocation."""
    return await get_mesh().invoke(domain, method, scent, session_id, payload)


# =============================================================================
# FASTAPI INTEGRATION
# =============================================================================

def add_mesh_routes(app, local_domains: List[str]):
    """
    Add mesh routes to a FastAPI app with fallback support.

    Usage:
        from brain_mesh import add_mesh_routes
        add_mesh_routes(app, ["flesh", "self"])
    """
    from pydantic import BaseModel

    class InvokeRequest(BaseModel):
        domain: str
        method: str
        scent: str
        session_id: str = ""
        payload: dict = {}

    @app.post("/ada/invoke")
    async def ada_invoke(request: InvokeRequest):
        mesh = get_mesh()

        # Local domain handling
        if request.domain in local_domains:
            return {"_local": True, "domain": request.domain}

        # Route with fallback support
        return await mesh.invoke(
            request.domain,
            request.method,
            request.scent,
            request.session_id,
            request.payload,
        )

    @app.get("/brain/topology")
    async def get_topology():
        return get_mesh().get_topology()

    @app.get("/brain/status")
    async def get_status():
        return get_mesh().get_status()

    @app.get("/brain/health")
    async def check_all_health():
        mesh = get_mesh()
        health = {"primary": {}, "fallback": {}}

        for name in PRIMARY_NODES:
            health["primary"][name] = await mesh._check_node_health(PRIMARY_NODES[name])

        for name in FALLBACK_NODES:
            health["fallback"][name] = await mesh._check_node_health(FALLBACK_NODES[name])

        return health

    @app.post("/brain/mode/{mode}")
    async def set_architecture_mode(mode: str):
        try:
            new_mode = ArchitectureMode(mode)
            set_mode(new_mode)
            return {"mode": new_mode.value, "status": "updated"}
        except ValueError:
            return {"error": f"Invalid mode: {mode}. Use: full, minimal, auto"}

    @app.post("/brain/reset-fallback")
    async def reset_fallback():
        """Reset fallback state, try all primary nodes again."""
        mesh = get_mesh()
        mesh._using_fallback.clear()
        for node in PRIMARY_NODES.values():
            node.fail_count = 0
        return {"status": "reset", "all_domains_on_primary": True}
