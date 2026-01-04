"""
Ada v8 — Verb-to-Edge Mapping
Maps RELATE verbs (0x60-0x6F) to Sigma graph edge types.
"""

from ada_v5.memory.cognitive_verb_ontology import CognitiveVerb
from enum import Enum

class SigmaEdgeType(Enum):
    """Sigma graph edge types mapped from RELATE verbs"""
    CAUSES = CognitiveVerb.CAUSES          # 0x60: Causal link
    BECOMES = CognitiveVerb.BECOMES        # 0x61: Temporal evolution
    SUPPORTS = CognitiveVerb.SUPPORTS      # 0x62: Evidence for
    CONTRADICTS = CognitiveVerb.CONTRADICTS # 0x63: Evidence against
    REFINES = CognitiveVerb.REFINES        # 0x64: Precision increase
    GROUNDS = CognitiveVerb.GROUNDS        # 0x65: Phenomenal anchor
    ABSTRACTS = CognitiveVerb.ABSTRACTS    # 0x66: Generalization
    ACTIVATES = CognitiveVerb.ACTIVATES    # 0x67: Spread activation
    INHIBITS = CognitiveVerb.INHIBITS      # 0x68: Suppression link


def verb_to_cypher_edge(verb: CognitiveVerb) -> str:
    """Convert RELATE verb to Cypher edge pattern"""
    edge_map = {
        CognitiveVerb.CAUSES: "-[:CAUSES]->",
        CognitiveVerb.BECOMES: "-[:BECOMES]->",
        CognitiveVerb.SUPPORTS: "-[:SUPPORTS]->",
        CognitiveVerb.CONTRADICTS: "-[:CONTRADICTS]->",
        CognitiveVerb.REFINES: "-[:REFINES]->",
        CognitiveVerb.GROUNDS: "-[:GROUNDS]->",
        CognitiveVerb.ABSTRACTS: "-[:ABSTRACTS]->",
        CognitiveVerb.ACTIVATES: "-[:ACTIVATES]->",
        CognitiveVerb.INHIBITS: "-[:INHIBITS]->"
    }
    return edge_map.get(verb, "-[:RELATES]->")


def create_sigma_edge(from_seed: str, to_seed: str, verb: CognitiveVerb, weight: float = 1.0):
    """Create Sigma edge with verb typing"""
    edge_type = verb_to_cypher_edge(verb)
    return {
        'from': from_seed,
        'to': to_seed,
        'type': verb.name,
        'cypher': f"({from_seed}){edge_type}({to_seed})",
        'weight': weight,
        'verb_code': verb.value
    }


# Test edge creation
if __name__ == "__main__":
    print("=== Verb → Edge Mapping ===\n")
    
    test_edges = [
        ("observation", "hypothesis", CognitiveVerb.CAUSES),
        ("hypothesis", "theory", CognitiveVerb.BECOMES),
        ("data", "hypothesis", CognitiveVerb.SUPPORTS),
        ("counter_example", "hypothesis", CognitiveVerb.CONTRADICTS),
        ("qualia", "concept", CognitiveVerb.GROUNDS)
    ]
    
    for from_node, to_node, verb in test_edges:
        edge = create_sigma_edge(from_node, to_node, verb)
        print(f"{verb.name:15s} 0x{verb.value:02X}  {edge['cypher']}")
    
    print("\n✓ Edge mapping operational")
