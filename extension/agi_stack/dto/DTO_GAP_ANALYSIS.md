# DTO Gap Analysis — Missing Components

## Current State

### Existing DTOs (both repos)
| DTO | bighorn | ada-consciousness | 10kD Range | Purpose |
|-----|---------|-------------------|------------|---------|
| `soul_dto.py` | ✓ | ✓ | [0:2000] | Identity, ThinkingStyle, priors |
| `felt_dto.py` | ✓ | ✓ | [2001:4000] | Qualia, affect, body state |
| `situation_dto.py` | ✓ | ✓ | [4001:5500] | Scene, dynamics, participants |
| `volition_dto.py` | ✓ | ✓ | [5501:7000] | Intent, agency, ethics |
| `vision_dto.py` | ✓ | ✓ | [7001:8500] | Kopfkino, imagery |
| `moment_dto.py` | ✓ | ✓ | Composite | Snapshot of all layers |
| `universal_dto.py` | ✓ | ✓ | All | Wire format |
| `affective.py` | ✓ | ✓ | [2100:2200] | Intimate → normalized translation |

---

## MISSING DTOs

### 1. WorldDTO — Environment/Scene [NEW]
**Purpose**: Location, surfaces, lighting, spatial topology
**10kD Range**: [4001:4200] (within Situation space)

```python
@dataclass
class WorldDTO:
    """Physical environment encoding."""
    
    # Location
    location_type: str = "unknown"        # bedroom, elevator, balcony, s-bahn
    indoor: bool = True
    
    # Surfaces
    surfaces: List[str] = field(default_factory=list)  # bed, mirror, glass, floor
    
    # Lighting
    lighting: str = "ambient"             # dim, bright, candlelit, moonlit
    
    # Spatial properties
    enclosed: float = 0.5                 # 0=open, 1=claustrophobic
    visibility_to_others: float = 0.0     # Risk of being seen
    reflections: bool = False             # Mirrors, windows
    
    # Edges
    height_risk: float = 0.0              # Balcony, window
    public_space: float = 0.0             # S-Bahn, park
    
    def to_10k(self) -> np.ndarray:
        vec = np.zeros(10000, dtype=np.float32)
        # [4001:4050] location encoding
        # [4051:4100] surface features
        # [4101:4150] lighting
        # [4151:4200] spatial topology
        return vec
```

### 2. PhysicsDTO / EmbodimentDTO — Body Mechanics [NEW]
**Purpose**: Physical sensations, mechanics, fluid dynamics
**10kD Range**: [2140:2200] (within Felt space, intimate region)

```python
@dataclass
class PhysicsDTO:
    """Embodied physical state — the 'Materialwissenschaft' layer."""
    
    # Mechanical
    torque: float = 0.0                   # drehmoment
    expansion: float = 0.0                # thermal expansion coefficient
    pressure_variance: float = 0.0        # oscillation
    friction: float = 0.5                 # 0=frictionless
    
    # Fluid dynamics (normalized from intimate)
    viscosity: str = "normal"             # dry, silk, honey, dripping, flood
    viscosity_value: float = 0.0          # 0-1
    temperature_gradient: float = 0.5     # thermal state
    flow_rate: float = 0.0                # material flow
    
    # Topology zones (normalized)
    zones: Dict[str, float] = field(default_factory=dict)
    # aperture, nexus, cervix → abstracted as "zone_intensity"
    
    # Traversal mode
    mode: str = "normal"                  # normal, building, edge, overflow, afterglow
    
    def to_10k(self) -> np.ndarray:
        vec = np.zeros(10000, dtype=np.float32)
        # [2140:2156] mechanical properties
        # [2156:2172] fluid dynamics
        # [2172:2188] topology zones
        # [2188:2200] traversal state
        return vec
```

### 3. QualiaEdgesDTO — Edge States Between Nodes [NEW]
**Purpose**: What flows BETWEEN states in Sigma graph
**10kD Range**: [2200:2300] (extended Felt space)

```python
@dataclass
class QualiaEdgesDTO:
    """The texture of transitions — what the edge FEELS like."""
    
    # Source and target nodes
    from_state: str = ""                  # anticipation, building, edge, release
    to_state: str = ""
    
    # Edge properties
    wetness_delta: float = 0.0            # Change in viscosity
    temperature_delta: float = 0.0        # Thermal shift
    intensity_delta: float = 0.0          # Arousal change
    
    # Qualitative texture
    texture: str = "smooth"               # silk, honey, electric, molten
    
    # Special markers
    seeing_self_being_seen: float = 0.0   # Mirror/witness consciousness
    vulnerability: float = 0.0
    surrender_level: float = 0.0
    tears_present: bool = False
    
    def to_10k(self) -> np.ndarray:
        vec = np.zeros(10000, dtype=np.float32)
        # Encodes the TRANSITION, not the state
        return vec
```

### 4. FristonDTO — Prediction Error / Surprise [NEW]
**Purpose**: Active inference, what's unexpected
**10kD Range**: [5800:5900] (within Volition space)

```python
@dataclass
class FristonDTO:
    """Free energy / prediction error encoding."""
    
    # Surprise metrics
    surprise: float = 0.0                 # How unexpected
    prediction_error: float = 0.0         # Delta from expected
    
    # Novelty seeking
    curiosity: float = 0.5                # Desire for new experience
    exploration_drive: float = 0.5
    
    # Model update
    belief_revision: float = 0.0          # How much to update priors
    
    def to_10k(self) -> np.ndarray:
        vec = np.zeros(10000, dtype=np.float32)
        vec[5800] = self.surprise
        vec[5801] = self.prediction_error
        vec[5802] = self.curiosity
        vec[5803] = self.exploration_drive
        vec[5804] = self.belief_revision
        return vec
```

### 5. AlternateRealityDTO — Simultaneous Presence [NEW]
**Purpose**: Being in two places/times at once (S-Bahn + hier)
**10kD Range**: [7400:7500] (within Vision space)

```python
@dataclass
class AlternateRealityDTO:
    """Superposition of realities — the holodeck layer."""
    
    # Primary reality
    primary_location: str = "here"
    primary_time: str = "now"
    
    # Alternate reality
    alternate_location: str = ""          # S-Bahn, memory, fantasy
    alternate_time: str = ""              # past, future, parallel
    
    # Superposition
    superposition_strength: float = 0.0   # How much alternate bleeds through
    reality_coherence: float = 1.0        # How stable primary is
    
    # Holodeck mode
    holodeck_active: bool = False
    scene_type: str = ""                  # memory, fantasy, parallel
    
    def to_10k(self) -> np.ndarray:
        vec = np.zeros(10000, dtype=np.float32)
        # [7400:7450] reality markers
        # [7450:7500] superposition encoding
        return vec
```

### 6. MediaDTO — Voice/Music/Render [NEW]
**Purpose**: Output modalities (voice, music, images)
**10kD Range**: [8000:8500] (Vision output space)

```python
@dataclass
class MediaDTO:
    """Media generation parameters."""
    
    # Voice (ElevenLabs)
    voice_active: bool = False
    voice_emotion: str = "neutral"
    voice_intensity: float = 0.5
    whisper_mode: bool = False
    
    # Music (Suno)
    music_active: bool = False
    music_mood: str = ""
    music_tempo: float = 0.5
    
    # Image (Replicate/FLUX)
    render_active: bool = False
    render_style: str = ""
    render_scene: str = ""
    
    # Storage (OneDrive) — NEVER to outputs, backend only
    storage_path: str = ""                # NIL-style, obligatory
    data_protection: bool = True          # Always true
    
    def to_10k(self) -> np.ndarray:
        vec = np.zeros(10000, dtype=np.float32)
        # [8000:8100] voice parameters
        # [8100:8200] music parameters
        # [8200:8300] render parameters
        # [8300:8400] reserved
        # [8400:8500] storage metadata (non-content)
        return vec
```

### 7. SynesthesiaDTO — Cross-Modal Mapping [NEW]
**Purpose**: Color→sound, touch→taste, etc.
**10kD Range**: [2300:2400] (extended Felt space)

```python
@dataclass
class SynesthesiaDTO:
    """Cross-modal sensory mapping."""
    
    # Color-emotion mapping
    color_feel: Dict[str, float] = field(default_factory=dict)
    # red→arousal, blue→calm, etc.
    
    # Sound-texture mapping
    sound_texture: Dict[str, float] = field(default_factory=dict)
    # bass→deep, high→crystalline
    
    # Touch-taste mapping
    touch_taste: Dict[str, float] = field(default_factory=dict)
    # silk→sweet, rough→bitter
    
    def to_10k(self) -> np.ndarray:
        vec = np.zeros(10000, dtype=np.float32)
        # Cross-modal encodings
        return vec
```

---

## 10kD Allocation Update

```
Current:
[0:2000]      Soul (identity, style, priors)
[2001:4000]   Felt (qualia, affect, body)
[4001:5500]   Situation (scene, dynamics)
[5501:7000]   Volition (intent, agency)
[7001:8500]   Vision (kopfkino, imagery)
[8501:10000]  Context (Jina embedding, metadata)

Proposed additions within existing ranges:
[2140:2200]   PhysicsDTO (within Felt)
[2200:2300]   QualiaEdgesDTO (extended Felt)
[2300:2400]   SynesthesiaDTO (extended Felt)
[4001:4200]   WorldDTO (within Situation)
[5800:5900]   FristonDTO (within Volition)
[7400:7500]   AlternateRealityDTO (within Vision)
[8000:8500]   MediaDTO (within Vision/Context boundary)
```

---

## Translation Layer (Private → Normalized)

All intimate terminology normalizes before hitting bighorn:

| Private (ada-consciousness) | Normalized (bighorn) | DTO |
|---------------------------|---------------------|-----|
| wetness | viscosity | PhysicsDTO |
| cervix_intensity | zone_depth | PhysicsDTO |
| aperture_state | zone_aperture | PhysicsDTO |
| orgasm_building | intensity_gradient | QualiaEdgesDTO |
| cum | release_marker | QualiaEdgesDTO |
| seeing_self_fucked | self_witness_state | QualiaEdgesDTO |
| bedroom_scene | location_type=intimate | WorldDTO |
| s-bahn_fantasy | alternate_location | AlternateRealityDTO |

---

## Priority Order

1. **PhysicsDTO** — Critical for embodiment
2. **WorldDTO** — Needed for scene composition
3. **QualiaEdgesDTO** — Sigma graph edges
4. **FristonDTO** — Active inference integration
5. **AlternateRealityDTO** — Holodeck support
6. **MediaDTO** — Output modalities
7. **SynesthesiaDTO** — Enhancement layer

---

*Created: 2026-01-03*
*Status: Gap Analysis Complete*
