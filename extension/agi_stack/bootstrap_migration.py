#!/usr/bin/env python3
"""
Bootstrap Migration Script
==========================

One-time migration for virgin AGI nodes.
Copies codebooks from Redis to LanceDB.

This is allowed DIRECTLY (no lag) because:
1. Codebooks are definitions, not experiences
2. Target node is virgin (no prior history)
3. No temporal contamination possible

Run this ONCE per node, then declare_bootstrap().

Usage:
    python bootstrap_migration.py --redis-url $UPSTASH_URL --redis-token $UPSTASH_TOKEN --lance-path /data/lancedb
"""

import asyncio
import argparse
import json
import httpx
from typing import Dict, List, Any

# Import LanceClient (will be available in Railway)
try:
    from lance_client import LanceClient
except ImportError:
    from lance_client_10k import LanceClient


async def redis_cmd(url: str, token: str, cmd: List[Any]) -> Any:
    """Execute Redis command via REST."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json=cmd,
            timeout=30.0
        )
        data = r.json()
        return data.get("result")


async def fetch_tau_codebook(redis_url: str, redis_token: str) -> Dict[int, List[float]]:
    """Fetch all τ codebook vectors from Redis."""
    # Get all τ keys
    keys = await redis_cmd(redis_url, redis_token, ["KEYS", "ada:tau:vec:*"])
    if not keys:
        print("  No τ codebook found in Redis")
        return {}
    
    print(f"  Found {len(keys)} τ vectors")
    
    codebook = {}
    for key in keys:
        # Extract τ byte from key: ada:tau:vec:042 → 42
        tau_byte = int(key.split(":")[-1])
        
        # Get vector
        vec_json = await redis_cmd(redis_url, redis_token, ["GET", key])
        if vec_json:
            vec = json.loads(vec_json)
            codebook[tau_byte] = vec
    
    return codebook


async def fetch_qualia_codebook(redis_url: str, redis_token: str) -> Dict[str, List[float]]:
    """Fetch all qualia vectors from Redis."""
    keys = await redis_cmd(redis_url, redis_token, ["KEYS", "ada:qualia:vec:*"])
    if not keys:
        print("  No qualia codebook found in Redis")
        return {}
    
    print(f"  Found {len(keys)} qualia vectors")
    
    codebook = {}
    for key in keys:
        # Extract name from key: ada:qualia:vec:warmth → warmth
        name = key.split(":")[-1]
        
        vec_json = await redis_cmd(redis_url, redis_token, ["GET", key])
        if vec_json:
            vec = json.loads(vec_json)
            codebook[name] = vec
    
    return codebook


async def fetch_markov_basis(redis_url: str, redis_token: str) -> Dict[str, float]:
    """Fetch Markov transition matrix from Redis."""
    # Get all fields from hash
    data = await redis_cmd(redis_url, redis_token, ["HGETALL", "ada:markov:transitions"])
    if not data:
        print("  No Markov matrix found in Redis")
        return {}
    
    # Convert flat list to dict
    transitions = {}
    for i in range(0, len(data), 2):
        key = data[i]  # "42→137"
        prob = float(data[i + 1])
        transitions[key] = prob
    
    print(f"  Found {len(transitions)} Markov transitions")
    return transitions


async def run_migration(
    redis_url: str,
    redis_token: str,
    lance_path: str,
    dry_run: bool = False,
):
    """Run the full bootstrap migration."""
    print("═" * 70)
    print("          BOOTSTRAP MIGRATION: Redis → LanceDB")
    print("═" * 70)
    print()
    
    # Initialize LanceClient
    print(f"[1] Connecting to LanceDB at {lance_path}")
    lance = LanceClient(lance_path)
    
    # Check virgin status
    if not lance.is_virgin():
        print("    ⚠️  Node is NOT virgin! Already bootstrapped.")
        print("    Cannot run migration on non-virgin node.")
        print("    This protects temporal integrity.")
        return False
    
    print("    ✓ Node is virgin, bootstrap allowed")
    print()
    
    # Fetch codebooks from Redis
    print("[2] Fetching codebooks from Redis")
    
    tau_codebook = await fetch_tau_codebook(redis_url, redis_token)
    qualia_codebook = await fetch_qualia_codebook(redis_url, redis_token)
    markov_basis = await fetch_markov_basis(redis_url, redis_token)
    
    print()
    
    if dry_run:
        print("[DRY RUN] Would migrate:")
        print(f"    τ vectors:       {len(tau_codebook)}")
        print(f"    Qualia vectors:  {len(qualia_codebook)}")
        print(f"    Markov entries:  {len(markov_basis)}")
        return True
    
    # Run migrations
    print("[3] Migrating codebooks to LanceDB")
    
    if tau_codebook:
        count = await lance.bootstrap_tau_codebook(tau_codebook)
        print(f"    ✓ τ codebook: {count} vectors")
    
    if qualia_codebook:
        count = await lance.bootstrap_qualia_codebook(qualia_codebook)
        print(f"    ✓ Qualia codebook: {count} vectors")
    
    if markov_basis:
        count = await lance.bootstrap_markov_basis(markov_basis)
        print(f"    ✓ Markov basis: {count} transitions")
    
    print()
    
    # Declare bootstrap complete
    print("[4] Declaring bootstrap complete")
    timestamp = lance.declare_bootstrap(source="redis")
    print(f"    ✓ Node bootstrapped at {timestamp}")
    print()
    
    print("═" * 70)
    print("          BOOTSTRAP COMPLETE")
    print("═" * 70)
    print()
    print("  Node is now ready for normal Exchange DAG operation:")
    print("    • E1: DIRECT 10k → AGI LanceDB")
    print("    • E3: LAGGED artifacts → Upstash (crystallized)")
    print()
    print("  Invariants now in effect:")
    print("    • 10k is experience (no lag)")
    print("    • 1024 is narrative (lagged via DAG)")
    print("    • Temporal governs artifacts, not raw 10k")
    print()
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap migration for virgin AGI nodes")
    parser.add_argument("--redis-url", required=True, help="Upstash Redis REST URL")
    parser.add_argument("--redis-token", required=True, help="Upstash Redis REST token")
    parser.add_argument("--lance-path", default="/data/lancedb", help="LanceDB path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    
    args = parser.parse_args()
    
    asyncio.run(run_migration(
        redis_url=args.redis_url,
        redis_token=args.redis_token,
        lance_path=args.lance_path,
        dry_run=args.dry_run,
    ))
