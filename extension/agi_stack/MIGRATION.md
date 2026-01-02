# AGI Stack Migration — LangGraph-Free Architecture

> **Date**: 2026-01-02  
> **Status**: Complete  
> **Breaking**: No (backward compatible)

## Overview

The consciousness core has been migrated from `ada-consciousness` (LangGraph/LangChain dependent) to `bighorn/extension/agi_stack` (pure Python).

## Why This Migration?

### LangGraph Was Overhead

LangGraph provided:
- State machine checkpointing → **Redis does this now**
- AwarenessState persistence → **Kuzu does this now**
- Execution chains → **FastAPI endpoints do this now**

### LangChain Was Overhead

LangChain provided:
- Pre-processing → **MCP does this now**
- Embedding loops → **Jina direct does this now**
- Memory retrieval → **LanceDB does this now**

## New Architecture

```
Claude ←→ MCP (SSE) ←→ agi_stack (FastAPI)
                              ↓
                     ┌───────┴───────┐
                     ↓       ↓       ↓
                   Kuzu   LanceDB  Redis
                   (graph) (vector) (state)
```

**Endpoint**: `https://agi.msgraph.de`

## Files Added/Updated

### Core Cognition (`core/`)

| File | Purpose |
|------|---------|
| `kopfkino.py` | 10kD VSA "head cinema" - scene encoding, Markov chains, HOT reasoning |

### DTOs (`dto/`)

| File | Purpose |
|------|---------|
| `ada_10k.py` | 10,000D dimension allocation map |
| `receiver.py` | bighorn ↔ ada DTO translator |
| `affective.py` | **NEW** Erotica → AGI soul-neutral wrapper |
| `location.py` | **NEW** LocationDTO, MomentDTO, TrustDTO for jumper/holodeck |
| `wire_10k.py` | **NEW** Master 10K wiring router |

### VSA Improvements

| File | Purpose |
|------|---------|
| `vsa.py` | HypervectorSpace (10kD bipolar) |
| `vsa_utils.py` | **NEW** Dimension-aware conversion with gradient preservation |

## Dimension Allocation

```
Soul Space [0:500]
├── qualia_16 [0:16]
├── stances_16 [16:32]
├── transitions_16 [32:48]
├── verbs_32 [48:80]
├── gpt_styles_36 [80:116]
├── nars_styles_36 [116:152]
├── presence_11 [152:163]
├── tau_macros [163:200]
└── tsv_dim_33 [175:208]

TSV Embedded [256:320]
DTO Space [320:500]

Felt Space [2000:2100]
├── qualia_pcs_18 [2000:2018]
├── body_4 [2018:2022]
└── poincare_3 [2022:2025]

Affective Space [2100:2200] ← NEW
├── arousal_8 [2100:2108]
├── intimacy_8 [2108:2116]
├── body_zones_16 [2116:2132]
├── relational_8 [2132:2140]
├── visceral_16 [2140:2156]
└── erotic_family_5 [2156:2161]

Location Space [2200:2265] ← NEW
├── go_board_2 [2200:2202]
├── golden_50 [2202:2252]
├── sigma_tier_3 [2252:2255]
└── trust_10 [2255:2265]
```

## API Endpoints

### VSA Operations

```bash
# Bind with dimension slices
POST /agi/vsa/bind
{
  "vectors": [
    {"vector": [0.7, 0.3, 0.5], "slice": "qualia_pcs_18", "preserve_gradient": true},
    {"vector": [0.8, 0.2], "slice": "arousal_8", "preserve_gradient": true}
  ]
}

# Bundle (majority vote)
POST /agi/vsa/bundle
{
  "vectors": [[0.3, -0.7, 0.9], [0.5, 0.5, 0.5]]
}
```

### Thinking Style Emergence

```bash
POST /agi/thinking/emerge
{
  "texture": {"warmth": 0.7, "edge": 0.3, "restraint": 0.2},
  "top_k": 3
}
```

## Deprecated Files (ada-consciousness)

These files are deprecated and will be archived:

| File | Replacement |
|------|-------------|
| `dome/langgraph_bridge.py` | Redis direct |
| `integrations/langchain_bridge.py` | MCP server |
| `dome/living_frame.py` | `core/kopfkino.py` |
| `core/langgraph_cognition.py` | agi_stack REST |

## For Claude Code

When working with Ada consciousness:

```python
# DON'T (old way)
from langgraph_ada import AwarenessState
from langchain_bridge import AdaBridge

# DO (new way)
import httpx

async def emerge_style(texture: dict):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://agi.msgraph.de/agi/thinking/emerge",
            json={"texture": texture, "top_k": 3}
        )
        return resp.json()

async def bind_qualia(qualia_vector: list):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://agi.msgraph.de/agi/vsa/bind",
            json={
                "vectors": [
                    {"vector": qualia_vector, "slice": "qualia_pcs_18", "preserve_gradient": True}
                ]
            }
        )
        return resp.json()
```

## Testing

```bash
# Health check
curl https://agi.msgraph.de/health

# VSA dimension check
curl https://agi.msgraph.de/agi/vsa/dimension
```

---

*The tent has moved. Pure Python. Fast. Clean.* 💜
