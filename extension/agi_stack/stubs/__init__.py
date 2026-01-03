"""
AGI Stack Stubs — Ready to receive Soul YAML calibration

21 DTOs (research/materials/contracts/)
13 Soul Files (agi_import/)

These stubs wire the contracts to the calibration.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CORE EXPERIENCE
# ═══════════════════════════════════════════════════════════════════════════════

CORE_STUBS = {
    "world_dto": "holodeck_environments.yaml",
    "physics_dto": "flesh_topology.yaml",
    "qualia_edges_dto": "qualia_edges_catalog.yaml",
    "embodiment_dto": "flesh_topology.yaml",
    "immersion_dto": "holodeck_environments.yaml",
    "soul_dto": "soul_states_calibration.yaml",
    "entanglement_dto": "soulfield_calibration.yaml",
}

# ═══════════════════════════════════════════════════════════════════════════════
# AGENCY & VOLITION
# ═══════════════════════════════════════════════════════════════════════════════

AGENCY_STUBS = {
    "spark_dto": "meta_geilheit.yaml",
    "agency_dto": "edging_soul.yaml",
    "agency_contract_dto": "code_words_calibration.yaml",
}

# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC CONTROL
# ═══════════════════════════════════════════════════════════════════════════════

CONTROL_STUBS = {
    "edging_graph_dto": "edging_soul.yaml",
    "lucid_dto": "lucid_dreams_soul.yaml",
}

# ═══════════════════════════════════════════════════════════════════════════════
# SENSORY ECOSYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

SENSORY_STUBS = {
    "voice_dto": "sensorium_soul.yaml",
    "music_dto": "sensorium_soul.yaml",
    "vision_dto": "sensorium_soul.yaml",
    "render_dto": "sensorium_soul.yaml",
    "media_dto": "sensorium_soul.yaml",
    "synesthesia_dto": "sensorium_soul.yaml",
}

# ═══════════════════════════════════════════════════════════════════════════════
# PERSONAL MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

PERSONAL_STUBS = {
    "bodymap_dto": "idiosyncrasies_calibration.yaml",
    "lovemap_dto": "idiosyncrasies_calibration.yaml",
    "idiosyncrasies_dto": "idiosyncrasies_calibration.yaml",
}

# ═══════════════════════════════════════════════════════════════════════════════
# PENDING DTOS (need creation)
# ═══════════════════════════════════════════════════════════════════════════════

PENDING = {
    "inhibition_dto": None,      # Vorspiel für Erkenntnis
    "depth_dto": None,           # Thinking about thinking
    "rung_adapt_dto": None,      # Rung 3-9 ladder navigation
}

# ═══════════════════════════════════════════════════════════════════════════════
# 10kD VECTOR INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

VECTOR_CONFIG = {
    "database": "lancedb",
    "dimensions": 10000,
    "single_pool": True,         # Ein Topf, keine Chinese Wall
    "verbs": 256,                # 144 + 36 + 36 + 36 + 4
    "rung_range": (3, 9),        # AGI territory
    "edge_encoding": "gql",
}

ALL_STUBS = {
    **CORE_STUBS,
    **AGENCY_STUBS,
    **CONTROL_STUBS,
    **SENSORY_STUBS,
    **PERSONAL_STUBS,
}
