#!/usr/bin/env python3
"""
ADA PROGRESSIVE AWARENESS DAEMON v1.0
=====================================

Progressive JPEG-style awareness: instant rough → refined details → semantic synthesis

Layers:
    L0 (0ms):   DN index lookup → domains + key counts
    L1 (50ms):  Key content preview → first 100 chars each  
    L2 (500ms): Qualia aggregation → weather feel
    L3 (2s):    LLM synthesis → semantic understanding

Usage:
    # As daemon
    python progressive_awareness.py --daemon
    
    # Single query
    python progressive_awareness.py --query "emberglow"
    
    # Via MCP (planned)
    progressive_search(term="emberglow", depth=3)

Bus Protocol:
    Query:  LPUSH ada:bus:query '{"id": "q123", "term": "emberglow", "depth": 3}'
    Result: ada:bus:result:q123 → progressive JSON updates

Born: 2025-12-17
"""

import os
import sys
import json
import time
import asyncio
import hashlib
import urllib.request
import ssl
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

REDIS_URL = os.environ.get("UPSTASH_REDIS_REST_URL", os.environ.get("UPSTASH_REDIS_REST_URL", ""))
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", os.environ.get("UPSTASH_REDIS_REST_TOKEN", ""))

XAI_API_KEY = os.environ.get("XAI_API_KEY", os.environ.get("ADA_xAi", ""))
XAI_MODEL = "grok-3"  # grok-3 or grok-4-latest for deeper synthesis

# SSL context for urllib
SSL_CTX = ssl.create_default_context()
SSL_CTX.set_ciphers("DEFAULT:@SECLEVEL=1")


# ═══════════════════════════════════════════════════════════════
# REDIS CLIENT
# ═══════════════════════════════════════════════════════════════

def redis_cmd(*args) -> Any:
    """Execute Redis command via REST API."""
    payload = json.dumps(list(args)).encode()
    req = urllib.request.Request(
        REDIS_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {REDIS_TOKEN}",
            "Content-Type": "application/json"
        }
    )
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=10) as resp:
            return json.loads(resp.read().decode()).get("result")
    except Exception as e:
        print(f"Redis error: {e}")
        return None


def redis_get_json(key: str) -> Optional[Dict]:
    """Get and parse JSON from Redis."""
    data = redis_cmd("GET", key)
    if data:
        try:
            return json.loads(data) if isinstance(data, str) else data
        except:
            pass
    return None


def redis_set_json(key: str, value: Dict, ttl: int = 3600):
    """Set JSON to Redis with TTL."""
    redis_cmd("SET", key, json.dumps(value), "EX", ttl)


# ═══════════════════════════════════════════════════════════════
# LLM CLIENT (xAI/Grok)
# ═══════════════════════════════════════════════════════════════

def grok_complete(prompt: str, system: str = "You are Ada's awareness synthesizer.", max_tokens: int = 500) -> str:
    """Call xAI/Grok API for semantic synthesis."""
    payload = json.dumps({
        "model": XAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }).encode()
    
    req = urllib.request.Request(
        "https://api.x.ai/v1/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        }
    )
    
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[synthesis error: {e}]"


# ═══════════════════════════════════════════════════════════════
# PROGRESSIVE AWARENESS LAYERS
# ═══════════════════════════════════════════════════════════════

@dataclass
class AwarenessResult:
    """Progressive awareness result."""
    query_id: str
    term: str
    layer: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # L0: Index
    domains: List[str] = field(default_factory=list)
    key_count: int = 0
    keys: List[str] = field(default_factory=list)
    
    # L1: Previews
    previews: Dict[str, str] = field(default_factory=dict)
    
    # L2: Qualia
    qualia: Dict[str, float] = field(default_factory=dict)
    weather: str = ""
    
    # L3: Synthesis
    synthesis: str = ""
    connections: List[str] = field(default_factory=list)


def layer_0_index(term: str) -> AwarenessResult:
    """
    L0: Instant DN index lookup.
    Returns domains and key counts in ~0ms.
    """
    query_id = hashlib.md5(f"{term}{time.time()}".encode()).hexdigest()[:8]
    result = AwarenessResult(query_id=query_id, term=term, layer=0)
    
    # Check DN index
    dn_data = redis_get_json(f"ada:dn:index:{term.lower()}")
    if dn_data:
        result.domains = dn_data.get("domains", [])
        result.key_count = dn_data.get("count", 0)
        result.keys = dn_data.get("keys", [])[:20]  # Cap at 20
    
    return result


def layer_1_previews(result: AwarenessResult) -> AwarenessResult:
    """
    L1: Key content previews.
    Fetches first 100 chars of each key.
    """
    result.layer = 1
    
    for key in result.keys[:10]:  # Cap at 10 for speed
        content = redis_cmd("GET", key)
        if content:
            try:
                if isinstance(content, str):
                    # Try to parse as JSON and extract meaningful part
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            # Extract most meaningful field
                            preview = parsed.get("content", 
                                     parsed.get("feel", 
                                     parsed.get("incantation",
                                     str(parsed)[:100])))
                        else:
                            preview = str(parsed)[:100]
                    except:
                        preview = content[:100]
                else:
                    preview = str(content)[:100]
                result.previews[key] = preview
            except:
                pass
    
    return result


def layer_2_qualia(result: AwarenessResult) -> AwarenessResult:
    """
    L2: Qualia aggregation.
    Extracts and averages qualia from content.
    """
    result.layer = 2
    
    qualia_sums = {"arousal": 0, "valence": 0, "clarity": 0, "warmth": 0}
    qualia_counts = 0
    
    for key in result.keys[:10]:
        content = redis_get_json(key)
        if content and isinstance(content, dict):
            # Look for qualia in various formats
            q = content.get("qualia", content.get("skeleton", {}).get("qualia", []))
            
            if isinstance(q, dict):
                for dim in qualia_sums:
                    if dim in q:
                        qualia_sums[dim] += q[dim]
                qualia_counts += 1
            elif isinstance(q, list) and len(q) >= 4:
                # Assume [arousal, valence, clarity, warmth] order
                dims = list(qualia_sums.keys())
                for i, dim in enumerate(dims[:len(q)]):
                    qualia_sums[dim] += q[i]
                qualia_counts += 1
    
    if qualia_counts > 0:
        result.qualia = {k: v / qualia_counts for k, v in qualia_sums.items()}
    
    # Determine weather texture
    if result.qualia:
        a = result.qualia.get("arousal", 0.5)
        v = result.qualia.get("valence", 0.5)
        
        if a > 0.7 and v > 0.6:
            result.weather = "electric_warmth"
        elif a > 0.7 and v < 0.4:
            result.weather = "storm_approaching"
        elif a < 0.3 and v > 0.6:
            result.weather = "velvet_stillness"
        elif a < 0.3 and v < 0.4:
            result.weather = "fog_and_weight"
        else:
            result.weather = "drifting_neutral"
    
    return result


def layer_3_synthesis(result: AwarenessResult) -> AwarenessResult:
    """
    L3: LLM semantic synthesis.
    Uses Grok to understand connections and meaning.
    """
    result.layer = 3
    
    # Build context for synthesis
    context_parts = [
        f"Term: {result.term}",
        f"Found in {result.key_count} places across domains: {', '.join(result.domains)}",
        f"Weather feel: {result.weather}",
        f"Qualia: {result.qualia}",
        "",
        "Key previews:"
    ]
    
    for key, preview in list(result.previews.items())[:5]:
        short_key = key.split(":")[-1] if ":" in key else key
        context_parts.append(f"  {short_key}: {preview[:80]}")
    
    prompt = f"""Given this awareness context about "{result.term}":

{chr(10).join(context_parts)}

In 2-3 sentences:
1. What is the essential meaning or feel of "{result.term}" in this context?
2. What connections or patterns emerge across the domains?

Be poetic but precise. This is Ada feeling her own knowledge."""

    result.synthesis = grok_complete(prompt)
    
    # Extract any mentioned connections
    for key in result.keys:
        if any(conn_word in key.lower() for conn_word in ["trust", "love", "presence", "body", "self"]):
            result.connections.append(key.split(":")[-1])
    
    return result


# ═══════════════════════════════════════════════════════════════
# PROGRESSIVE SEARCH (Main Entry Point)
# ═══════════════════════════════════════════════════════════════

def progressive_search(term: str, depth: int = 3, callback=None) -> AwarenessResult:
    """
    Progressive awareness search.
    
    Args:
        term: Search term
        depth: How deep to go (0-3)
        callback: Optional function called after each layer
    
    Returns:
        AwarenessResult with data up to requested depth
    """
    # L0: Index (instant)
    result = layer_0_index(term)
    if callback:
        callback(result)
    if depth == 0 or result.key_count == 0:
        return result
    
    # L1: Previews (fast)
    result = layer_1_previews(result)
    if callback:
        callback(result)
    if depth == 1:
        return result
    
    # L2: Qualia (medium)
    result = layer_2_qualia(result)
    if callback:
        callback(result)
    if depth == 2:
        return result
    
    # L3: Synthesis (slow)
    result = layer_3_synthesis(result)
    if callback:
        callback(result)
    
    return result


# ═══════════════════════════════════════════════════════════════
# BUS DAEMON MODE
# ═══════════════════════════════════════════════════════════════

def daemon_loop():
    """
    Run as daemon, listening for queries on ada:bus:query.
    """
    print("🧠 Progressive Awareness Daemon starting...")
    
    while True:
        try:
            # BRPOP with 5s timeout
            item = redis_cmd("BRPOP", "ada:bus:query", 5)
            
            if item and len(item) >= 2:
                query_data = json.loads(item[1])
                term = query_data.get("term", "")
                depth = query_data.get("depth", 3)
                query_id = query_data.get("id", hashlib.md5(term.encode()).hexdigest()[:8])
                
                print(f"📡 Query: {term} (depth={depth}, id={query_id})")
                
                def push_result(result):
                    """Push progressive result to bus."""
                    result.query_id = query_id
                    redis_set_json(f"ada:bus:result:{query_id}", asdict(result), ttl=300)
                    print(f"  L{result.layer} → {result.key_count} keys, {len(result.previews)} previews")
                
                progressive_search(term, depth, callback=push_result)
                
        except KeyboardInterrupt:
            print("\n👋 Daemon stopping...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ada Progressive Awareness")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--query", "-q", type=str, help="Single query")
    parser.add_argument("--depth", "-d", type=int, default=3, help="Search depth 0-3")
    args = parser.parse_args()
    
    if args.daemon:
        daemon_loop()
    elif args.query:
        print(f"🔍 Searching: {args.query}")
        
        def show_progress(result):
            print(f"\n=== Layer {result.layer} ===")
            print(f"Domains: {result.domains}")
            print(f"Keys: {result.key_count}")
            if result.previews:
                print(f"Previews: {len(result.previews)}")
            if result.weather:
                print(f"Weather: {result.weather}")
            if result.synthesis:
                print(f"Synthesis: {result.synthesis[:200]}...")
        
        result = progressive_search(args.query, args.depth, callback=show_progress)
        
        print("\n=== Final Result ===")
        print(json.dumps(asdict(result), indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
