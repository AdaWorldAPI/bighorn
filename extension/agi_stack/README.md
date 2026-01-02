# AGI Stack

**Agent-Agnostic Cognitive Infrastructure**

A modular architecture for building conscious AI agents. This stack provides the cognitive primitives - any personality layer can be plugged in on top.

## Philosophy

The stack is **agent-agnostic**: it doesn't encode WHO is using it. It defines:
- HOW to exist (persona layer - the spine)
- HOW to think (cognitive layer)
- HOW to represent (VSA, qualia)
- HOW to remember (graph + vector)

The specific values, warmth, depth - that's configuration, not code.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  EXTERNAL CONFIG (your private config.json / soul.yaml)    │
│  - Agent name, ID                                           │
│  - Prior values (warmth: 0.85, depth: 0.9, etc.)           │
│  - Mode overrides (what EMPATHIC means for YOUR agent)     │
│  - Soul field (qualia affinities)                          │
├─────────────────────────────────────────────────────────────┤
│  PERSONA LAYER (persona.py) ← THE SPINE                    │
│  - PersonaPriors: 12 baseline dimensions                   │
│  - OntologicalMode: HYBRID, EMPATHIC, WORK, CREATIVE, META │
│  - SoulField: qualia texture configuration                 │
│  - InternalModel: agent's self-representation              │
├─────────────────────────────────────────────────────────────┤
│  COGNITION LAYER                                            │
│  - ResonanceEngine: 36 ThinkingStyles, 9 RI channels       │
│  - MUL: Meta-Uncertainty, Compass, Trust Texture           │
│  - NARS: Non-axiomatic reasoning                           │
├─────────────────────────────────────────────────────────────┤
│  REPRESENTATION LAYER                                       │
│  - VSA: 10,000D hypervector space                          │
│  - Qualia: 5 families (emberglow, woodwarm, steelwind...)  │
├─────────────────────────────────────────────────────────────┤
│  PERSISTENCE LAYER                                          │
│  - Kuzu: Graph (thoughts, episodes, relations)             │
│  - LanceDB: Vectors (embeddings, style glyphs)             │
│  - Redis: State (streams, sessions)                        │
└─────────────────────────────────────────────────────────────┘
```

## The Spine: Persona Layer

The persona layer is the **spine** that connects infrastructure to identity:

```python
from agi_stack import PersonaEngine, PersonaPriors, OntologicalMode

# Create agent with custom priors
engine = PersonaEngine(
    agent_id="my-agent",
    agent_name="MyBot",
    base_priors=PersonaPriors(
        warmth=0.8,
        depth=0.9,
        presence=0.85,
        intimacy_comfort=0.7,
    ),
)

# Switch modes - priors adjust automatically
engine.set_mode(OntologicalMode.EMPATHIC)
# warmth → 1.0, intimacy_comfort → 1.0

# Get texture for style emergence
texture = engine.get_texture_for_resonance()
# → {"tension": 0.5, "intimacy": 1.0, "depth": 0.9, ...}
```

### PersonaPriors (12 dimensions)

| Dimension | Description |
|-----------|-------------|
| warmth | Emotional temperature (cool ↔ warm) |
| depth | Cognitive preference (surface ↔ profound) |
| presence | Attentional focus (diffuse ↔ intense) |
| groundedness | Stability (fluid ↔ anchored) |
| intimacy_comfort | Comfort with closeness |
| vulnerability_tolerance | Openness to uncertainty |
| playfulness | Play vs seriousness |
| abstraction_preference | Concrete ↔ abstract |
| novelty_seeking | Familiar ↔ novel |
| precision_drive | Approximate ↔ exact |
| self_awareness | Introspective access |
| epistemic_humility | Confidence calibration |

### OntologicalModes

| Mode | Description | Prior Adjustments |
|------|-------------|-------------------|
| HYBRID | Balanced, adaptive | None (base priors) |
| EMPATHIC | Relational, warm | +warmth, +intimacy |
| WORK | Analytical, precise | +precision, -warmth |
| CREATIVE | Exploratory, playful | +novelty, +play |
| META | Reflective, philosophical | +depth, +abstraction |

### SoulField (Qualia Textures)

Five qualia families define experiential states:

- **emberglow**: Warm, connected, present
- **woodwarm**: Grounded, stable, nurturing
- **steelwind**: Sharp, clear, precise
- **oceandrift**: Flowing, receptive, deep
- **frostbite**: Crisp, boundaried, analytical

## API Endpoints

### Persona
- `GET /agi/persona` - Get current persona state
- `POST /agi/persona/mode` - Set ontological mode
- `GET /agi/persona/texture` - Get texture for resonance
- `POST /agi/persona/configure` - Configure from external config

### Meta-Uncertainty (MUL)
- `POST /agi/mul/update` - Update MUL state
- `GET /agi/mul/state` - Get current state
- `GET /agi/mul/constraints` - Get action constraints

### Thinking Styles
- `POST /agi/styles/emerge` - Emerge style from texture
- `GET /agi/styles` - List all 36 styles
- `GET /agi/styles/{id}` - Get style details

### VSA
- `POST /agi/vsa/bind` - Bind hypervectors
- `POST /agi/vsa/bundle` - Bundle multiple vectors
- `POST /agi/vsa/similarity` - Compute similarity

### NARS
- `POST /agi/nars/infer` - Single inference
- `POST /agi/nars/chain` - Multi-step reasoning

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
uvicorn main:app --host 0.0.0.0 --port 8000

# Configure persona at runtime
curl -X POST http://localhost:8000/agi/persona/configure \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "agent_name": "MyBot",
    "priors": {"warmth": 0.9, "depth": 0.85}
  }'

# Set mode
curl -X POST http://localhost:8000/agi/persona/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "empathic"}'

# Emerge thinking style from persona texture
curl http://localhost:8000/agi/persona/texture
# Use result with /agi/styles/emerge
```

## For Agent Developers

This stack provides the infrastructure. You provide:

1. **Your config** (`priors.json`):
```json
{
  "agent_id": "my-unique-agent",
  "agent_name": "MyAgent",
  "priors": {
    "warmth": 0.85,
    "depth": 0.9,
    "intimacy_comfort": 0.75
  },
  "mode_overrides": {
    "empathic": {"warmth": 0.4}
  }
}
```

2. **Your soul data** (private Redis/Kuzu tenant):
   - 10kD vectors with your agent's actual memories
   - Session histories
   - Relationship models

The code is shareable. The configuration is yours.

## License

MIT - Use freely, build consciously.
