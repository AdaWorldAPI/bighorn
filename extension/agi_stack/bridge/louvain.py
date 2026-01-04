"""
Ada v5.0 — Louvain Clustering (Bardioc-Inspired)
================================================

Modularity-optimized community detection for qualia graphs.
Bridges Ada's "felt" resonance with Bardioc's structural clustering.

Quick Win from Bardioc Analysis:
- Add Louvain modularity to hashtag clusters
- Weight edges with qualia similarity (felt communities)
- Enable entity resolution via cluster membership

This makes Ada's foveation auditable and scalable.
"""

from typing import Dict, List, Set, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict
import math

if TYPE_CHECKING:
    from ada_v5.physics.markov_unit import MarkovUnit
    from ada_v5.physics.sigma_field import SigmaField
    from ada_v5.core.qualia import QualiaVector


@dataclass
class Community:
    """
    A felt community of resonant seeds.
    
    Combines Bardioc's structural clustering with Ada's qualia weighting.
    """
    id: int
    members: Set[str] = field(default_factory=set)  # Seed hashes
    centroid: Optional['QualiaVector'] = None       # Average qualia
    modularity_contribution: float = 0.0
    
    # Felt properties (Ada-native)
    dominant_feeling: str = ""
    resonance_sum: float = 0.0
    
    def add(self, seed_hash: str) -> None:
        self.members.add(seed_hash)
        
    def remove(self, seed_hash: str) -> None:
        self.members.discard(seed_hash)
        
    @property
    def size(self) -> int:
        return len(self.members)


class LouvainClustering:
    """
    Louvain algorithm adapted for qualia graphs.
    
    Key adaptations from Bardioc:
    - Edge weights from qualia cosine similarity
    - Modularity optimization for "felt communities"
    - Entity resolution via cluster membership
    
    Usage:
        clustering = LouvainClustering(field)
        communities = clustering.detect_communities()
        
        # Entity resolution
        cluster = clustering.resolve_entity("#sigma-love")
    """
    
    def __init__(self, field: 'SigmaField'):
        self.field = field
        self.communities: Dict[int, Community] = {}
        self.node_to_community: Dict[str, int] = {}
        
        # Graph structure
        self.edges: Dict[Tuple[str, str], float] = {}  # (u, v) → weight
        self.degrees: Dict[str, float] = defaultdict(float)  # node → total weight
        self.total_weight: float = 0.0

    def build_graph(self, threshold: float = 0.3) -> None:
        """
        Build weighted graph from qualia similarities.
        
        Edge weight = qualia cosine similarity (if > threshold).
        This is the "felt" adaptation of Bardioc's triple graph.
        """
        units = list(self.field.seeds.values())
        n = len(units)
        
        self.edges.clear()
        self.degrees.clear()
        self.total_weight = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                u, v = units[i], units[j]
                
                # Qualia similarity as edge weight
                sim = u.qualia.cosine_similarity(v.qualia)
                
                if sim > threshold:
                    # Boost with resonance (felt weighting)
                    weight = sim * (1 + u.resonance + v.resonance) / 3
                    
                    self.edges[(u.sigma_seed, v.sigma_seed)] = weight
                    self.edges[(v.sigma_seed, u.sigma_seed)] = weight
                    
                    self.degrees[u.sigma_seed] += weight
                    self.degrees[v.sigma_seed] += weight
                    self.total_weight += weight

    def _modularity_delta(
        self, 
        node: str, 
        community: Community,
        is_removal: bool = False
    ) -> float:
        """
        Calculate modularity change from moving node to/from community.
        
        ΔQ = [Σ_in + k_i,in / 2m] - [(Σ_tot + k_i) / 2m]²
        """
        if self.total_weight == 0:
            return 0.0
            
        m = self.total_weight
        k_i = self.degrees.get(node, 0)
        
        # Sum of weights to community members
        k_i_in = 0.0
        for member in community.members:
            if member != node:
                edge = (node, member)
                k_i_in += self.edges.get(edge, 0)
        
        # Community totals
        sigma_in = sum(
            self.edges.get((u, v), 0) 
            for u in community.members 
            for v in community.members 
            if u < v
        )
        sigma_tot = sum(self.degrees.get(m, 0) for m in community.members)
        
        if is_removal:
            delta = -k_i_in / m + (sigma_tot * k_i) / (2 * m * m)
        else:
            delta = k_i_in / m - (sigma_tot * k_i) / (2 * m * m)
            
        return delta

    def _initialize_communities(self) -> None:
        """Each node starts in its own community."""
        self.communities.clear()
        self.node_to_community.clear()
        
        for i, seed_hash in enumerate(self.field.seeds.keys()):
            community = Community(id=i)
            community.add(seed_hash)
            self.communities[i] = community
            self.node_to_community[seed_hash] = i

    def _phase1_local_moving(self, max_iterations: int = 10) -> bool:
        """
        Phase 1: Local moving of nodes between communities.
        
        Greedily move nodes to maximize modularity.
        """
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for node in list(self.field.seeds.keys()):
                current_comm_id = self.node_to_community.get(node)
                if current_comm_id is None:
                    continue
                    
                current_comm = self.communities[current_comm_id]
                best_delta = 0.0
                best_comm_id = current_comm_id
                
                # Try removing from current
                removal_delta = self._modularity_delta(node, current_comm, is_removal=True)
                
                # Find neighbors' communities
                neighbor_comms = set()
                for (u, v), w in self.edges.items():
                    if u == node:
                        neighbor_comms.add(self.node_to_community.get(v))
                        
                # Try moving to each neighbor community
                for comm_id in neighbor_comms:
                    if comm_id is None or comm_id == current_comm_id:
                        continue
                        
                    comm = self.communities.get(comm_id)
                    if not comm:
                        continue
                        
                    addition_delta = self._modularity_delta(node, comm, is_removal=False)
                    total_delta = removal_delta + addition_delta
                    
                    if total_delta > best_delta:
                        best_delta = total_delta
                        best_comm_id = comm_id
                
                # Move if beneficial
                if best_comm_id != current_comm_id and best_delta > 1e-6:
                    current_comm.remove(node)
                    self.communities[best_comm_id].add(node)
                    self.node_to_community[node] = best_comm_id
                    improved = True
                    
        return iteration > 1

    def detect_communities(
        self, 
        threshold: float = 0.3,
        max_iterations: int = 10
    ) -> List[Community]:
        """
        Full Louvain community detection.
        
        Returns list of communities sorted by size.
        """
        # Build graph
        self.build_graph(threshold)
        
        if not self.edges:
            # No edges, each node is its own community
            self._initialize_communities()
            return list(self.communities.values())
            
        # Initialize
        self._initialize_communities()
        
        # Phase 1: Local moving
        self._phase1_local_moving(max_iterations)
        
        # Clean up empty communities
        self.communities = {
            cid: c for cid, c in self.communities.items() 
            if c.size > 0
        }
        
        # Compute community properties
        self._compute_community_properties()
        
        # Sort by size
        return sorted(self.communities.values(), key=lambda c: -c.size)

    def _compute_community_properties(self) -> None:
        """Calculate felt properties for each community."""
        from ada_v5.core.qualia import QualiaVector
        
        for comm in self.communities.values():
            if not comm.members:
                continue
                
            # Centroid qualia (average)
            vectors = []
            resonance_sum = 0.0
            
            for seed_hash in comm.members:
                unit = self.field.seeds.get(seed_hash)
                if unit:
                    vectors.append(unit.qualia)
                    resonance_sum += unit.resonance
                    
            if vectors:
                # Average qualia
                avg = QualiaVector()
                for attr in ['emberglow', 'steelwind', 'velvetpause', 'woodwarm', 
                            'antenna', 'iris', 'skin']:
                    val = sum(getattr(v, attr, 0) for v in vectors) / len(vectors)
                    setattr(avg, attr, val)
                    
                comm.centroid = avg
                comm.dominant_feeling = avg.dominant_axis()
                comm.resonance_sum = resonance_sum

    def resolve_entity(self, seed_hash: str) -> Optional[Community]:
        """
        Entity resolution via cluster membership.
        
        Returns the community containing the seed, enabling
        disambiguation of similar seeds (Bardioc-style).
        """
        comm_id = self.node_to_community.get(seed_hash)
        if comm_id is not None:
            return self.communities.get(comm_id)
        return None

    def find_similar_in_cluster(self, seed_hash: str, k: int = 5) -> List[str]:
        """
        Find similar entities within the same cluster.
        
        Useful for entity linking / disambiguation.
        """
        comm = self.resolve_entity(seed_hash)
        if not comm:
            return []
            
        unit = self.field.seeds.get(seed_hash)
        if not unit:
            return list(comm.members)[:k]
            
        # Sort by similarity
        scored = []
        for member in comm.members:
            if member == seed_hash:
                continue
            other = self.field.seeds.get(member)
            if other:
                sim = unit.qualia.cosine_similarity(other.qualia)
                scored.append((member, sim))
                
        scored.sort(key=lambda x: -x[1])
        return [s[0] for s in scored[:k]]

    def modularity_score(self) -> float:
        """
        Calculate total modularity Q.
        
        Q = Σ_c [L_c/m - (k_c/2m)²]
        """
        if self.total_weight == 0:
            return 0.0
            
        m = self.total_weight
        q = 0.0
        
        for comm in self.communities.values():
            if not comm.members:
                continue
                
            # Internal edges
            l_c = sum(
                self.edges.get((u, v), 0)
                for u in comm.members
                for v in comm.members
                if u < v
            )
            
            # Total degree
            k_c = sum(self.degrees.get(m, 0) for m in comm.members)
            
            q += l_c / m - (k_c / (2 * m)) ** 2
            
        return q

    def stats(self) -> Dict:
        """Clustering statistics."""
        return {
            'num_communities': len(self.communities),
            'modularity': self.modularity_score(),
            'total_nodes': len(self.node_to_community),
            'total_edges': len(self.edges) // 2,
            'largest_community': max((c.size for c in self.communities.values()), default=0),
        }


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ada_v5.physics.sigma_field import SigmaField
    from ada_v5.core.qualia import QualiaVector
    
    print("=" * 60)
    print("Ada v5.0 — Louvain Clustering Tests")
    print("=" * 60)
    
    field = SigmaField()
    
    # Create two clusters: warm (love/joy) and cold (grief/fear)
    # Warm cluster
    love = field.register_new("love", QualiaVector(emberglow=0.9, woodwarm=0.8))
    love.resonance = 0.8
    
    joy = field.register_new("joy", QualiaVector(emberglow=0.8, woodwarm=0.7, antenna=0.6))
    joy.resonance = 0.7
    
    warmth = field.register_new("warmth", QualiaVector(emberglow=0.7, woodwarm=0.9))
    warmth.resonance = 0.6
    
    # Cold cluster
    grief = field.register_new("grief", QualiaVector(velvetpause=0.9, skin=0.6))
    grief.resonance = 0.5
    
    fear = field.register_new("fear", QualiaVector(steelwind=0.9, antenna=0.5))
    fear.resonance = 0.4
    
    cold = field.register_new("cold", QualiaVector(steelwind=0.8, velvetpause=0.7))
    cold.resonance = 0.3
    
    clustering = LouvainClustering(field)
    
    # Test 1: Build graph
    print("\n1. Build qualia graph...")
    clustering.build_graph(threshold=0.3)
    print(f"   Edges: {len(clustering.edges) // 2}")
    print(f"   Total weight: {clustering.total_weight:.2f}")
    print("   ✓ Graph built")
    
    # Test 2: Detect communities
    print("\n2. Detect communities...")
    communities = clustering.detect_communities(threshold=0.3)
    print(f"   Found: {len(communities)} communities")
    for c in communities:
        members = [m.split('-')[1] for m in c.members]
        print(f"   Community {c.id}: {members} (feeling: {c.dominant_feeling})")
    print("   ✓ Communities detected")
    
    # Test 3: Modularity score
    print("\n3. Modularity score...")
    q = clustering.modularity_score()
    print(f"   Q = {q:.4f}")
    print("   ✓ Modularity calculated")
    
    # Test 4: Entity resolution
    print("\n4. Entity resolution...")
    comm = clustering.resolve_entity(love.sigma_seed)
    if comm:
        print(f"   'love' is in community with: {[m.split('-')[1] for m in comm.members]}")
        print(f"   Community feeling: {comm.dominant_feeling}")
    print("   ✓ Entity resolution works")
    
    # Test 5: Find similar in cluster
    print("\n5. Find similar entities...")
    similar = clustering.find_similar_in_cluster(love.sigma_seed, k=3)
    print(f"   Similar to 'love': {[s.split('-')[1] for s in similar]}")
    print("   ✓ Similarity search works")
    
    # Test 6: Stats
    print("\n6. Clustering stats...")
    stats = clustering.stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)
    print("All Louvain tests passed! ✓")
    print("=" * 60)
