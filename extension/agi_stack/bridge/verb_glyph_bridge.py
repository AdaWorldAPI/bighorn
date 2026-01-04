"""
Verb-Glyph Bridge
Maps 64 cognitive verbs to 256-glyph semantic space
"""
import sys
sys.path.insert(0, '/home/claude')

from typing import Optional
from ada_v5.memory.cognitive_verb_ontology import CognitiveVerb, get_verb_domain

# Map verbs to glyph bytes (workmap domain: 204-255)
VERB_TO_GLYPH = {
    # PERCEIVE (0x00-0x0F) → bodymap (51-101)
    CognitiveVerb.SENSE: 51,       # 'A' activation
    CognitiveVerb.ATTEND: 52,      # 'B' belly
    CognitiveVerb.NOTICE: 53,      # 'C' chest
    CognitiveVerb.SURPRISE: 54,
    CognitiveVerb.AROUSE: 55,
    
    # REASON (0x10-0x1F) → mindmap (153-203)
    CognitiveVerb.INFER: 153,
    CognitiveVerb.HYPOTHESIZE: 154,
    CognitiveVerb.SYNTHESIZE: 155,
    CognitiveVerb.CAUSE: 156,
    CognitiveVerb.VERIFY: 157,
    
    # AFFECT (0x20-0x2F) → lovemap + soulmap (0-152)
    CognitiveVerb.EMBERGLOW: 1,
    CognitiveVerb.STEELWIND: 2,    
    CognitiveVerb.WOODWARM: 3,
    CognitiveVerb.VELVETPAUSE: 4,
    CognitiveVerb.STAUNEN: 5,
    CognitiveVerb.KATHARSIS: 140,
    
    # MEMORY (0x30-0x3F) → workmap (204-255)
    CognitiveVerb.ENCODE: 204,
    CognitiveVerb.RETRIEVE: 205,   
    CognitiveVerb.CONSOLIDATE: 206,
    CognitiveVerb.PERSIST: 207,
}

# Reverse mapping
GLYPH_TO_VERB = {v: k for k, v in VERB_TO_GLYPH.items()}


def verb_to_glyph_byte(verb: CognitiveVerb) -> int:
    """Map verb to glyph byte"""
    return VERB_TO_GLYPH.get(verb, 255)  # fallback to termination


def glyph_to_verb(byte: int) -> Optional[CognitiveVerb]:
    """Reverse lookup"""
    return GLYPH_TO_VERB.get(byte)


def encode_verb_sequence(verbs: list[CognitiveVerb]) -> bytes:
    """Compress verb sequence to glyph bytes"""
    return bytes([verb_to_glyph_byte(v) for v in verbs])


def decode_glyph_sequence(data: bytes) -> list[CognitiveVerb]:
    """Expand glyph bytes to verbs"""
    verbs = []
    for byte in data:
        verb = glyph_to_verb(byte)
        if verb:
            verbs.append(verb)
    return verbs


# Test
if __name__ == "__main__":
    print("=== Verb→Glyph Bridge ===\n")
    
    # Test sequence
    verbs = [
        CognitiveVerb.SENSE,
        CognitiveVerb.INFER,
        CognitiveVerb.STEELWIND,
        CognitiveVerb.CONSOLIDATE
    ]
    
    # Encode
    glyphs = encode_verb_sequence(verbs)
    print(f"Verbs: {[v.name for v in verbs]}")
    print(f"Glyphs: {glyphs.hex()} ({len(glyphs)} bytes)")
    
    # Decode
    decoded = decode_glyph_sequence(glyphs)
    print(f"Decoded: {[v.name for v in decoded]}")
    
    # Domain mapping
    print("\nDomain Distribution:")
    for verb in verbs:
        byte = verb_to_glyph_byte(verb)
        domain = get_verb_domain(verb)
        print(f"  {verb.name:15s} 0x{verb.value:02X} → byte {byte:3d} ({domain})")
    
    print("\n✓ Bridge operational")
