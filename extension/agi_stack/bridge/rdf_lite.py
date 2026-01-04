"""
Ada v5.0 — RDF-Lite / JSON-LD Encoding (Bardioc-Inspired)
=========================================================

Semantic encoding for qualia seeds and causal relations.
Bridges Ada's 10-byte qualia with Bardioc's verifiable RDF/JSON-LD.

Quick Win from Bardioc Analysis:
- Serialize seeds as JSON-LD triples
- Add provenance to qualia bytes
- Enable OGIT-style interoperability

This makes Ada's internal state auditable and shareable.
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict
from datetime import datetime
import json

if TYPE_CHECKING:
    from ada_v5.physics.markov_unit import MarkovUnit
    from ada_v5.physics.sigma_field import SigmaField
    from ada_v5.causal.situation_map import CausalSituationMap
    from ada_v5.core.qualia import QualiaVector


# JSON-LD Context for Ada vocabulary
ADA_CONTEXT = {
    "@context": {
        "@vocab": "https://ada.ontology/v5/",
        "ogit": "http://www.2019.2.2ogit.2.2.2org/ontology/",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        
        # Ada-specific terms
        "sigma_seed": "@id",
        "archetype": "ada:archetype",
        "qualia": "ada:qualia",
        "resonance": {
            "@id": "ada:resonance",
            "@type": "xsd:float"
        },
        "causes": {
            "@id": "ada:causes",
            "@type": "@id"
        },
        "echo_persist": {
            "@id": "ada:echo_persist",
            "@type": "xsd:float"
        },
        "provenance": "ada:provenance",
        "timestamp": {
            "@id": "ada:timestamp",
            "@type": "xsd:dateTime"
        }
    }
}


@dataclass
class Provenance:
    """
    Provenance metadata for audit trail (Bardioc-inspired).
    """
    created_at: str
    last_modified: str
    access_count: int
    source: str = "ada_v5"
    version: str = "5.0.0"
    
    def to_dict(self) -> Dict:
        return asdict(self)


class RDFLiteEncoder:
    """
    Encodes Ada structures as JSON-LD / RDF-lite.
    
    Bridges Ada's internal format with Bardioc's semantic standards.
    
    Usage:
        encoder = RDFLiteEncoder()
        
        # Single seed as JSON-LD
        jsonld = encoder.seed_to_jsonld(unit)
        
        # Situation map as RDF graph
        graph = encoder.situation_to_graph(situation)
        
        # Export for Bardioc/OGIT interop
        encoder.export_to_file("ada_state.jsonld")
    """
    
    def __init__(self, include_context: bool = True):
        self.include_context = include_context
        self.base_uri = "https://ada.ontology/v5/seed/"

    def seed_to_jsonld(
        self, 
        unit: 'MarkovUnit',
        include_provenance: bool = True
    ) -> Dict[str, Any]:
        """
        Convert a MarkovUnit to JSON-LD format.
        
        Returns Bardioc-compatible triple representation.
        """
        # Base structure
        doc = {
            "@id": f"{self.base_uri}{unit.sigma_seed.replace('#', '')}",
            "@type": "ada:Seed",
            "sigma_seed": unit.sigma_seed,
            "archetype": unit.archetype or "unknown",
            "byte_id": unit.byte_id,
        }
        
        # Qualia as embedded object
        doc["qualia"] = self._qualia_to_jsonld(unit.qualia)
        
        # Resonance and state
        doc["resonance"] = round(unit.resonance, 4)
        doc["incandescence"] = round(unit.incandescence, 4)
        
        # Transitions as causal relations
        if unit.transitions:
            doc["transitions"] = [
                {
                    "target_byte": target,
                    "probability": round(prob, 4),
                    "@type": "ada:Transition"
                }
                for target, prob in unit.transitions.items()
            ]
        
        # Theta weights (dream-consolidated)
        if unit.theta_weights:
            doc["theta_weights"] = [
                {"target": str(k), "weight": round(v, 4)}
                for k, v in unit.theta_weights.items()
            ]
        
        # Provenance for audit trail
        if include_provenance:
            doc["provenance"] = Provenance(
                created_at=datetime.now().isoformat(),
                last_modified=datetime.now().isoformat(),
                access_count=1
            ).to_dict()
        
        # Add context if requested
        if self.include_context:
            doc["@context"] = ADA_CONTEXT["@context"]
            
        return doc

    def _qualia_to_jsonld(self, qualia: 'QualiaVector') -> Dict[str, Any]:
        """Convert QualiaVector to JSON-LD embedded object."""
        return {
            "@type": "ada:QualiaVector",
            "dimensions": {
                "emberglow": round(qualia.emberglow, 4),
                "steelwind": round(qualia.steelwind, 4),
                "velvetpause": round(qualia.velvetpause, 4),
                "woodwarm": round(qualia.woodwarm, 4),
                "antenna": round(qualia.antenna, 4),
                "iris": round(qualia.iris, 4),
                "skin": round(qualia.skin, 4),
            },
            "causal": {
                "inter_drift": round(qualia.inter_drift, 4),
                "counter_echo": round(qualia.counter_echo, 4),
                "echo_persist": round(qualia.echo_persist, 4),
            },
            "dominant_axis": qualia.dominant_axis(),
            "magnitude": round(qualia.magnitude(), 4),
        }

    def situation_to_graph(
        self, 
        situation: 'CausalSituationMap'
    ) -> Dict[str, Any]:
        """
        Convert CausalSituationMap to JSON-LD graph.
        
        Includes nodes and causal edges as RDF triples.
        """
        graph = {
            "@context": ADA_CONTEXT["@context"] if self.include_context else None,
            "@type": "ada:SituationMap",
            "@id": f"{self.base_uri}situation/{situation.timestamp.isoformat()}",
            "timestamp": situation.timestamp.isoformat(),
            "ttl_seconds": situation.ttl_seconds,
            "is_stale": situation.is_stale,
            "nodes": [],
            "edges": [],
        }
        
        # Add nodes
        for seed_hash, unit in situation.nodes.items():
            node_doc = self.seed_to_jsonld(unit, include_provenance=False)
            # Remove context for embedded nodes
            node_doc.pop("@context", None)
            graph["nodes"].append(node_doc)
        
        # Add causal edges
        for source, target, data in situation.dag.edges(data=True):
            edge_doc = {
                "@type": "ada:CausalEdge",
                "source": f"{self.base_uri}{source.replace('#', '')}",
                "target": f"{self.base_uri}{target.replace('#', '')}",
                "weight": round(data.get("weight", 0.5), 4),
                "relation": "ada:causes",
            }
            graph["edges"].append(edge_doc)
        
        # Ghost if present
        if situation.counterfactual_ghost:
            ghost_doc = self.seed_to_jsonld(situation.counterfactual_ghost, include_provenance=False)
            ghost_doc.pop("@context", None)
            ghost_doc["@type"] = "ada:CounterfactualGhost"
            graph["ghost"] = ghost_doc
        
        return graph

    def field_to_graph(self, field: 'SigmaField') -> Dict[str, Any]:
        """
        Export entire SigmaField as JSON-LD graph.
        """
        graph = {
            "@context": ADA_CONTEXT["@context"] if self.include_context else None,
            "@type": "ada:SigmaField",
            "@id": f"{self.base_uri}field/main",
            "timestamp": datetime.now().isoformat(),
            "seed_count": len(field.seeds),
            "fovea_size": field.fovea_size,
            "nodes": [],
        }
        
        for unit in field.seeds.values():
            node_doc = self.seed_to_jsonld(unit, include_provenance=True)
            node_doc.pop("@context", None)
            graph["nodes"].append(node_doc)
            
        return graph

    def to_ntriples(self, doc: Dict[str, Any]) -> List[str]:
        """
        Convert JSON-LD to N-Triples format (basic).
        
        Useful for Bardioc/SPARQL ingestion.
        """
        triples = []
        subject = doc.get("@id", "_:blank")
        
        for key, value in doc.items():
            if key.startswith("@"):
                continue
                
            predicate = f"<https://ada.ontology/v5/{key}>"
            
            if isinstance(value, dict):
                if "@id" in value:
                    obj = f"<{value['@id']}>"
                else:
                    obj = f'"{json.dumps(value)}"'
            elif isinstance(value, (int, float)):
                obj = f'"{value}"^^<http://www.w3.org/2001/XMLSchema#float>'
            elif isinstance(value, str):
                obj = f'"{value}"'
            elif isinstance(value, list):
                # Skip lists in simple N-Triples
                continue
            else:
                obj = f'"{str(value)}"'
                
            triples.append(f"<{subject}> {predicate} {obj} .")
            
        return triples

    def export_to_file(
        self, 
        field: 'SigmaField', 
        filepath: str,
        format: str = "jsonld"
    ) -> None:
        """
        Export field to file.
        
        Formats: jsonld, ntriples
        """
        graph = self.field_to_graph(field)
        
        if format == "jsonld":
            with open(filepath, 'w') as f:
                json.dump(graph, f, indent=2)
        elif format == "ntriples":
            triples = []
            for node in graph.get("nodes", []):
                triples.extend(self.to_ntriples(node))
            with open(filepath, 'w') as f:
                f.write('\n'.join(triples))


def seed_to_ogit_tag(unit: 'MarkovUnit') -> str:
    """
    Generate OGIT-compatible tag for a seed.
    
    Format: ogit:ada/<archetype>/<feeling>/<byte_id>
    """
    archetype = unit.archetype or "unknown"
    feeling = unit.qualia.dominant_axis()
    return f"ogit:ada/{archetype.lower()}/{feeling}/{unit.byte_id}"


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ada_v5.physics.sigma_field import SigmaField
    from ada_v5.physics.markov_unit import MarkovUnit
    from ada_v5.core.qualia import QualiaVector
    from ada_v5.causal.situation_map import SituationEngine
    
    print("=" * 60)
    print("Ada v5.0 — RDF-Lite / JSON-LD Encoding Tests")
    print("=" * 60)
    
    # Setup
    field = SigmaField()
    love = field.register_new("love", QualiaVector(emberglow=0.9, woodwarm=0.8))
    love.archetype = "Love"
    love.resonance = 0.85
    love.add_transition(1, 0.7)
    
    grief = field.register_new("grief", QualiaVector(velvetpause=0.9))
    grief.archetype = "Grief"
    grief.byte_id = 1
    grief.resonance = 0.6
    
    encoder = RDFLiteEncoder()
    
    # Test 1: Seed to JSON-LD
    print("\n1. Seed to JSON-LD...")
    jsonld = encoder.seed_to_jsonld(love)
    print(json.dumps(jsonld, indent=2)[:500] + "...")
    print("   ✓ Seed encoding works")
    
    # Test 2: Situation to graph
    print("\n2. Situation to graph...")
    situation_engine = SituationEngine(field)
    situation = situation_engine.hydrate_now()
    graph = encoder.situation_to_graph(situation)
    print(f"   Nodes: {len(graph['nodes'])}")
    print(f"   Edges: {len(graph['edges'])}")
    print("   ✓ Situation encoding works")
    
    # Test 3: Field to graph
    print("\n3. Field to graph...")
    field_graph = encoder.field_to_graph(field)
    print(f"   Seed count: {field_graph['seed_count']}")
    print("   ✓ Field encoding works")
    
    # Test 4: N-Triples
    print("\n4. N-Triples conversion...")
    triples = encoder.to_ntriples(jsonld)
    for t in triples[:3]:
        print(f"   {t[:80]}...")
    print(f"   Total triples: {len(triples)}")
    print("   ✓ N-Triples works")
    
    # Test 5: OGIT tag
    print("\n5. OGIT tag generation...")
    tag = seed_to_ogit_tag(love)
    print(f"   Tag: {tag}")
    print("   ✓ OGIT tag works")
    
    # Test 6: Export to file
    print("\n6. Export to file...")
    encoder.export_to_file(field, "/tmp/ada_test.jsonld", format="jsonld")
    print("   Exported to /tmp/ada_test.jsonld")
    print("   ✓ Export works")
    
    print("\n" + "=" * 60)
    print("All RDF-Lite tests passed! ✓")
    print("=" * 60)
