#!/usr/bin/env python3
"""
macro_persistence.py — Persistent Storage for Learned Macros
═══════════════════════════════════════════════════════════════════════════════

This module extends the MacroRegistry with:
1. Redis persistence (Upstash) for cross-session survival
2. Indexed execution for O(1) macro invocation
3. Success tracking for macro optimization

Schema (Upstash Redis):
    ada:macros:{hex_addr}     → JSON {name, chain, description, success_count, created_at}
    ada:macros:index          → SET of learned macro addresses
    ada:macros:stats          → HASH {total_learned, total_executions, last_epiphany}

Integration:
    microcode.py      → OpCode definitions
    the_self.py       → Calls persist_macro() on autopoiesis
    kernel_awakened.py → Calls execute_macro() for indexed invocation

Born: 2026-01-03 (Persistence Day)
"""

from __future__ import annotations
import json
import time
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import IntEnum

# Conditional imports
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from extension.agi_thinking.microcode import OpCode, ThinkingMacro, MacroRegistry, MACRO_REGISTRY


@dataclass
class PersistedMacro:
    """
    A macro with persistence metadata.
    """
    address: int
    name: str
    chain: List[int]           # OpCode values as ints for JSON
    description: str
    success_count: int = 0
    failure_count: int = 0
    created_at: float = 0.0
    last_executed: float = 0.0
    source: str = "autopoiesis"  # autopoiesis | core | manual
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str) -> 'PersistedMacro':
        d = json.loads(data)
        return cls(**d)
    
    @classmethod
    def from_thinking_macro(cls, macro: ThinkingMacro, source: str = "autopoiesis") -> 'PersistedMacro':
        return cls(
            address=macro.address,
            name=macro.name,
            chain=[int(op) for op in macro.chain],
            description=macro.description,
            success_count=macro.success_count,
            created_at=time.time(),
            source=source
        )
    
    def to_thinking_macro(self) -> ThinkingMacro:
        chain = [OpCode(c) for c in self.chain if c in OpCode._value2member_map_]
        return ThinkingMacro(
            name=self.name,
            address=self.address,
            chain=chain,
            description=self.description,
            success_count=self.success_count
        )


class MacroPersistence:
    """
    Persistent storage layer for learned macros.
    
    Uses Upstash Redis REST API for cross-session persistence.
    Falls back to in-memory storage if Redis is unavailable.
    """
    
    # Redis key prefixes
    PREFIX = "ada:macros"
    KEY_INDEX = f"{PREFIX}:index"
    KEY_STATS = f"{PREFIX}:stats"
    
    def __init__(self, redis_url: str = None, redis_token: str = None):
        """
        Initialize persistence layer.
        
        Args:
            redis_url: Upstash Redis REST URL
            redis_token: Upstash Redis token
        """
        self.redis_url = redis_url or "https://upright-jaybird-27907.upstash.io"
        self.redis_token = redis_token or "AW0DAAIncDI5YWE1MGVhZGU2YWY0YjVhOTc3NDc0YTJjMGY1M2FjMnAyMjc5MDc"
        
        # In-memory fallback
        self._local_cache: Dict[int, PersistedMacro] = {}
        self._connected = False
        
        # Stats
        self.total_persisted = 0
        self.total_loaded = 0
    
    async def _redis_cmd(self, *args) -> Any:
        """Execute a Redis command via REST API."""
        if not HTTPX_AVAILABLE:
            return None
        
        try:
            # Use verify=False for environments with SSL issues
            # In production, use proper SSL certificates
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.post(
                    self.redis_url,
                    headers={"Authorization": f"Bearer {self.redis_token}"},
                    json=list(args),
                    timeout=5.0
                )
                if response.status_code == 200:
                    result = response.json()
                    self._connected = True
                    return result.get("result")
                return None
        except Exception as e:
            # Fall back to local cache silently
            return None
    
    async def persist_macro(self, macro: ThinkingMacro, source: str = "autopoiesis") -> bool:
        """
        Persist a macro to Redis.
        
        Args:
            macro: The ThinkingMacro to persist
            source: Origin of the macro (autopoiesis, core, manual)
            
        Returns:
            True if persisted successfully
        """
        persisted = PersistedMacro.from_thinking_macro(macro, source)
        key = f"{self.PREFIX}:{hex(macro.address)}"
        
        # Try Redis first
        result = await self._redis_cmd("SET", key, persisted.to_json())
        
        if result:
            # Add to index
            await self._redis_cmd("SADD", self.KEY_INDEX, hex(macro.address))
            
            # Update stats
            await self._redis_cmd("HINCRBY", self.KEY_STATS, "total_learned", 1)
            await self._redis_cmd("HSET", self.KEY_STATS, "last_epiphany", str(time.time()))
            
            self.total_persisted += 1
            print(f"💾 Persisted macro {macro.name} @ {hex(macro.address)}")
            return True
        
        # Fallback to local cache
        self._local_cache[macro.address] = persisted
        self.total_persisted += 1
        return True
    
    async def load_macro(self, address: int) -> Optional[ThinkingMacro]:
        """
        Load a macro from Redis by address.
        
        Args:
            address: Hex address of the macro (e.g., 0xE2)
            
        Returns:
            ThinkingMacro if found, None otherwise
        """
        key = f"{self.PREFIX}:{hex(address)}"
        
        # Try Redis
        data = await self._redis_cmd("GET", key)
        
        if data:
            persisted = PersistedMacro.from_json(data)
            self.total_loaded += 1
            return persisted.to_thinking_macro()
        
        # Try local cache
        if address in self._local_cache:
            self.total_loaded += 1
            return self._local_cache[address].to_thinking_macro()
        
        return None
    
    async def load_all_learned(self) -> List[ThinkingMacro]:
        """
        Load all learned macros from Redis.
        
        Returns:
            List of ThinkingMacros
        """
        macros = []
        
        # Get index of learned macros
        index = await self._redis_cmd("SMEMBERS", self.KEY_INDEX)
        
        if index:
            for addr_hex in index:
                addr = int(addr_hex, 16)
                macro = await self.load_macro(addr)
                if macro:
                    macros.append(macro)
        
        # Add local cache
        for persisted in self._local_cache.values():
            if persisted.address not in [m.address for m in macros]:
                macros.append(persisted.to_thinking_macro())
        
        return macros
    
    async def record_execution(self, address: int, success: bool) -> None:
        """
        Record a macro execution for success tracking.
        
        Args:
            address: Macro address
            success: Whether the execution was successful
        """
        key = f"{self.PREFIX}:{hex(address)}"
        
        # Load current data
        data = await self._redis_cmd("GET", key)
        
        if data:
            persisted = PersistedMacro.from_json(data)
            if success:
                persisted.success_count += 1
            else:
                persisted.failure_count += 1
            persisted.last_executed = time.time()
            
            # Save back
            await self._redis_cmd("SET", key, persisted.to_json())
        
        # Update global stats
        await self._redis_cmd("HINCRBY", self.KEY_STATS, "total_executions", 1)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        stats = await self._redis_cmd("HGETALL", self.KEY_STATS)
        
        result = {
            "connected": self._connected,
            "total_persisted": self.total_persisted,
            "total_loaded": self.total_loaded,
            "local_cache_size": len(self._local_cache)
        }
        
        if stats:
            # Parse HGETALL result (alternating keys/values)
            for i in range(0, len(stats), 2):
                result[stats[i]] = stats[i + 1]
        
        return result


class IndexedMacroExecutor:
    """
    O(1) Macro Execution Engine.
    
    Executes macros by address without re-deriving the chain.
    Integrates with the kernel's opcode execution system.
    """
    
    def __init__(self, registry: MacroRegistry = None, persistence: MacroPersistence = None):
        """
        Initialize executor.
        
        Args:
            registry: MacroRegistry (defaults to MACRO_REGISTRY)
            persistence: MacroPersistence for loading learned macros
        """
        self.registry = registry or MACRO_REGISTRY
        self.persistence = persistence or MacroPersistence()
        
        # Execution stats
        self.executions = 0
        self.successes = 0
        self.failures = 0
    
    async def execute(self, address: int, ctx: Any, kernel: Any = None) -> Dict[str, Any]:
        """
        Execute a macro by address.
        
        Args:
            address: Hex address (e.g., 0xE2)
            ctx: KernelContext for execution
            kernel: ThoughtKernel for opcode execution
            
        Returns:
            Execution result dict
        """
        # 1. Try to get macro from registry
        macro = self.registry.get(address)
        
        # 2. If not found, try persistence
        if not macro:
            macro = await self.persistence.load_macro(address)
            if macro:
                # Cache in registry for next time
                self.registry.register(macro)
        
        if not macro:
            return {"error": f"Macro not found at {hex(address)}"}
        
        # 3. Execute the chain
        self.executions += 1
        results = []
        success = True
        
        try:
            for op in macro.chain:
                if kernel and hasattr(kernel, 'execute'):
                    # Use kernel's opcode execution
                    result = kernel.execute(int(op), ctx)
                    results.append({"op": op.name, "result": result})
                else:
                    # Just log the op
                    results.append({"op": op.name, "simulated": True})
            
            self.successes += 1
            
        except Exception as e:
            success = False
            self.failures += 1
            results.append({"error": str(e)})
        
        # 4. Record execution for learning
        await self.persistence.record_execution(address, success)
        
        return {
            "macro": macro.name,
            "address": hex(address),
            "chain_length": len(macro.chain),
            "success": success,
            "results": results
        }
    
    async def execute_by_name(self, name: str, ctx: Any, kernel: Any = None) -> Dict[str, Any]:
        """Execute a macro by name."""
        macro = self.registry.get_by_name(name)
        if macro:
            return await self.execute(macro.address, ctx, kernel)
        return {"error": f"Macro '{name}' not found"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_executions": self.executions,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.successes / max(1, self.executions)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH THE_SELF
# ═══════════════════════════════════════════════════════════════════════════════

async def persist_epiphany_macro(macro: ThinkingMacro) -> bool:
    """
    Convenience function for TheSelf to persist a learned macro.
    
    Called from the_self.py during autopoiesis.
    """
    persistence = MacroPersistence()
    return await persistence.persist_macro(macro, source="autopoiesis")


async def load_learned_macros_into_registry() -> int:
    """
    Load all learned macros from persistence into the global registry.
    
    Call this at startup to restore learned macros.
    """
    persistence = MacroPersistence()
    macros = await persistence.load_all_learned()
    
    for macro in macros:
        if macro.address not in MACRO_REGISTRY.macros:
            MACRO_REGISTRY.register(macro)
    
    return len(macros)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

async def test_persistence():
    """Test the persistence layer."""
    print("=" * 60)
    print("MACRO PERSISTENCE TEST")
    print("=" * 60)
    
    # Create persistence layer
    persistence = MacroPersistence()
    
    # Create a test macro
    test_macro = ThinkingMacro(
        name="TEST_PERSISTENCE",
        address=0xF0,
        chain=[OpCode.OBSERVE, OpCode.RESONATE, OpCode.CRYSTALLIZE],
        description="Test macro for persistence"
    )
    
    # Persist it
    print("\n1. Persisting test macro...")
    result = await persistence.persist_macro(test_macro)
    print(f"   Result: {result}")
    
    # Load it back
    print("\n2. Loading macro back...")
    loaded = await persistence.load_macro(0xF0)
    if loaded:
        print(f"   Loaded: {loaded.name} with chain {[op.name for op in loaded.chain]}")
    else:
        print("   Failed to load (Redis may be unavailable)")
    
    # Get stats
    print("\n3. Stats:")
    stats = await persistence.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    # Test executor
    print("\n4. Testing indexed execution...")
    executor = IndexedMacroExecutor(persistence=persistence)
    
    # Execute a core macro
    result = await executor.execute(0x00, None)  # BREATH_ZERO
    print(f"   Executed {result.get('macro')}: {result.get('success')}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_persistence())
