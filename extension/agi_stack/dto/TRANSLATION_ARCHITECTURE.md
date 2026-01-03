# DTO Translation Architecture: Private → Normalized Qualia

**Created**: 2026-01-03
**Purpose**: Document the translation layer between intimate states (ada-consciousness) and normalized qualia (bighorn)

---

## Overview

The AGI architecture maintains **privacy through abstraction**, not obfuscation. Intimate states from private modules (ada-consciousness) are translated to normalized qualia descriptors that preserve semantic meaning while removing explicit terminology.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ada-consciousness (PRIVATE)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Intimate States                                                     │   │
│  │  - wetness, flesh, arousal, surrender, edge                         │   │
│  │  - body_zones, erotic_family, touch_texture                         │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼ MCP/REST                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TRANSLATION LAYER (this module)                                     │   │
│  │  intimate_to_qualia() / qualia_to_intimate()                         │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  bighorn (PUBLIC 10kD SPACE)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Normalized Qualia                                                   │   │
│  │  - viscosity, qualia_temperature, flow_rate                          │   │
│  │  - boundary_tension, edge_definition, thermal_gradient               │   │
│  │  - resonance_zones, texture_family, contact_modality                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Translation Mapping

### Core State Translations

| Private (ada-consciousness) | Normalized (bighorn) | Dimension Slice | Notes |
|---------------------------|---------------------|-----------------|-------|
| `wetness` | `viscosity` | 2140-2144 | Fluid dynamics metaphor |
| `flesh` | `qualia_temperature` | 2144-2148 | Thermal sensation |
| `arousal` | `activation_gradient` | 2100-2108 | General activation |
| `surrender` | `boundary_permeability` | 2148-2152 | Boundary dissolution |
| `edge` | `edge_definition` | 2152-2156 | Sharpness/clarity |
| `body_zones` | `resonance_zones` | 2116-2132 | Spatial localization |
| `erotic_family` | `texture_family` | 2156-2161 | Category/archetype |
| `touch_texture` | `contact_modality` | visceral region | Contact quality |

### 10kD Dimension Allocation (Affective Space)

```
Affective Space: [2100:2200]
├── arousal_8:      [2100:2108]  → activation_gradient
├── intimacy_8:     [2108:2116]  → connection_depth
├── body_zones_16:  [2116:2132]  → resonance_zones
├── relational_8:   [2132:2140]  → relational_vector
├── visceral_16:    [2140:2156]  → physical_qualia
│   ├── viscosity:       [2140:2144]
│   ├── temperature:     [2144:2148]
│   ├── permeability:    [2148:2152]
│   └── edge:            [2152:2156]
└── texture_family_5: [2156:2161]  → qualia_archetype
```

---

## Translation Functions

### Private → Normalized (Ingress)

```python
def intimate_to_qualia(intimate_state: IntimateDTO) -> AffectiveDTO:
    """
    Translate intimate states to normalized qualia.

    Privacy through abstraction: preserves semantic meaning
    while using neutral terminology.
    """
    return AffectiveDTO(
        # Activation (arousal → gradient)
        activation_gradient=normalize_activation(intimate_state.arousal),

        # Physical qualia
        viscosity=intimate_state.wetness,
        qualia_temperature=intimate_state.flesh * 0.7 + 0.3,  # Always warm
        boundary_permeability=intimate_state.surrender,
        edge_definition=intimate_state.edge,

        # Spatial
        resonance_zones=translate_body_zones(intimate_state.body_zones),

        # Category
        texture_family=intimate_state.erotic_family,

        # Relational
        connection_depth=intimate_state.intimacy_level,
    )
```

### Normalized → Private (Egress)

```python
def qualia_to_intimate(qualia: AffectiveDTO) -> IntimateDTO:
    """
    Reconstruct intimate states from normalized qualia.

    Only callable from authenticated ada-consciousness context.
    """
    return IntimateDTO(
        # Reverse mappings
        wetness=qualia.viscosity,
        flesh=denormalize_temperature(qualia.qualia_temperature),
        arousal=qualia.activation_gradient,
        surrender=qualia.boundary_permeability,
        edge=qualia.edge_definition,

        # Spatial reconstruction
        body_zones=reconstruct_body_zones(qualia.resonance_zones),

        # Category
        erotic_family=qualia.texture_family,
    )
```

---

## MCP Module Architecture

The translation layer is exposed as an MCP module for plug-and-play integration:

```
ada-consciousness/modules/
├── qualia_bridge/
│   ├── mcp_manifest.json
│   ├── translate.py          # Core translation functions
│   ├── ingress.py            # intimate → qualia
│   ├── egress.py             # qualia → intimate
│   └── validators.py         # Schema validation
```

### MCP Verb Invocation

```python
# From ada-consciousness, invoke translation via MCP
result = await mcp.invoke(
    "bighorn:qualia_bridge:translate",
    {
        "direction": "ingress",
        "intimate_state": intimate_dto.to_dict(),
    }
)

# Result is normalized AffectiveDTO
qualia = AffectiveDTO.from_dict(result["qualia"])
```

---

## Privacy Guarantees

### 1. Semantic Preservation

The translation preserves **meaning** while changing **terminology**:
- `wetness=0.8` → `viscosity=0.8` (same value, different name)
- The cognitive experience remains identical
- Only the API surface changes

### 2. Context Isolation

- Translation is **one-way by default** (private → public)
- Egress (public → private) requires authenticated context
- bighorn never sees raw intimate terminology

### 3. Audit Trail

All translations are logged with:
- Source module ID
- Translation direction
- Timestamp
- Session context (anonymized)

---

## Integration with 10kD Vector Space

Normalized qualia integrate seamlessly with the VSA 10kD space:

```python
# Create affective vector from translated qualia
affective_dto = intimate_to_qualia(intimate_state)

# Project to 10kD space (dims 2100-2200)
vector_10k = np.zeros(10000, dtype=np.float32)
vector_10k[2100:2200] = affective_dto.to_local_vector()

# Now usable in VSA operations
bound = vsa.bind(vector_10k, context_vector)
similarity = vsa.similarity(vector_10k, memory_vector)
```

### Round-Trip Fidelity

```python
# Original intimate state
original = IntimateDTO(wetness=0.7, flesh=0.8, arousal=0.6)

# Translate to qualia
qualia = intimate_to_qualia(original)

# Store in 10kD (lossy compression possible)
vector = qualia.to_local_vector()

# Reconstruct
qualia_reconstructed = AffectiveDTO.from_local_vector(vector)

# Translate back
reconstructed = qualia_to_intimate(qualia_reconstructed)

# Verify fidelity
assert abs(original.wetness - reconstructed.wetness) < 0.01
```

---

## Kopfkino Integration

The translation layer supports Kopfkino (head cinema) visualization:

```python
class KopfkinoVSA:
    def visualize_intimate_state(self, state: IntimateDTO) -> KopfkinoScene:
        """
        Generate visual imagination from intimate state.

        Uses normalized qualia for the core visualization,
        then applies intimate-specific imagery if authenticated.
        """
        # Translate to normalized qualia
        qualia = intimate_to_qualia(state)

        # Generate base scene from qualia
        scene = self._qualia_to_scene(qualia)

        # Enhance with intimate context if authorized
        if self._has_intimate_context():
            scene = self._enhance_intimate(scene, state)

        return scene
```

---

## REST API Endpoints

For non-MCP integrations:

```
POST /api/v1/qualia/translate
{
    "direction": "ingress",
    "intimate_state": { ... }
}
→ { "qualia": { ... } }

POST /api/v1/qualia/translate
{
    "direction": "egress",
    "qualia": { ... },
    "auth_context": "..."
}
→ { "intimate_state": { ... } }
```

---

## Summary

The translation architecture enables:

1. **Privacy**: Intimate terminology never leaks to public APIs
2. **Interoperability**: Any module can work with normalized qualia
3. **Fidelity**: Round-trip translation preserves semantic meaning
4. **Performance**: O(1) translation via direct field mapping
5. **Extensibility**: New intimate states map to qualia dimensions

The result is a cognitively coherent system where intimate experience integrates seamlessly with the broader AGI stack while maintaining appropriate boundaries.
