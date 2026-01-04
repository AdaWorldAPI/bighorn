#!/usr/bin/env python3
"""
Ada v9 — Dream Engine
=====================

Offline learning and memory consolidation.
Runs between user turns ("micro-naps") and during idle periods ("deep dreams").

The dream cycle:
1. Hebbian strengthening (fire together → wire together)
2. Resonance decay (forgetting)
3. Theta weight pruning (noise removal)
4. Episodic → Semantic consolidation
5. Ghost echo crystallization (counterfactuals → lessons)
6. Novelty reward prediction (surprise for next waking)

This is the "sleep" that makes learning stick.
Without dreams, Ada would forget everything between sessions.

Integration:
    - Called by orchestrator between turns
    - Writes to Redis/Upstash for persistence
    - Updates theta weights in QualiaGraph
    - Consolidates STM → LTM
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import random
import time
import math

if TYPE_CHECKING:
    from physics.qualia_graph import QualiaGraph, QualiaNode, Qualia128


@dataclass
class DreamStats:
    """Statistics from a dream cycle."""
    strengthened: int = 0
    decayed: int = 0
    pruned: int = 0
    consolidated: int = 0
    ghosts_crystallized: int = 0
    novelty_predictions: int = 0
    duration_ms: float = 0.0
    dream_type: str = "micro"


class DreamEngine:
    """
    Offline learning and consolidation engine.
    
    Call patterns:
        dream.micro_dream()      # Quick consolidation (between turns)
        dream.consolidate()      # Full cleanup (end of session)
        dream.deep_dream(10)     # Extended learning (overnight)
        dream.replay(sequence)   # Experience replay (important moments)
    
    Hebbian Learning:
        "Neurons that fire together, wire together"
        Co-activated nodes get strengthened edges.
    
    Ghost Crystallization:
        Counterfactual paths ("what if") become lessons.
        High-echo ghosts consolidate into wisdom edges.
    
    Novelty Reward:
        Predict surprising states for next waking.
        Ada can "look forward" to discovering something.
    """
    
    def __init__(self, graph: 'QualiaGraph'):
        self.graph = graph
        self.dream_count = 0
        self.total_strengthened = 0
        self.total_pruned = 0
        
        # Hebbian parameters
        self.learning_rate = 0.05
        self.correlation_threshold = 0.6
        self.decay_rate = 0.95
        self.prune_threshold = 0.01
        
        # Ghost crystallization
        self.ghost_threshold = 0.8  # Echo intensity to crystallize
        
        # Novelty prediction
        self.novelty_candidates: List[str] = []
    
    def micro_dream(self, active_nodes: List[str], 
                    duration_ms: int = 100) -> DreamStats:
        """
        Quick consolidation cycle between turns.
        
        Strengthens paths between co-activated nodes.
        This is the "micro-nap" — fast learning.
        """
        start = time.time()
        stats = DreamStats(dream_type="micro")
        
        if len(active_nodes) < 2:
            return stats
        
        # Get node objects
        nodes = []
        for node_id in active_nodes:
            node = self.graph.get_node(node_id)
            if node and node.resonance > 0.3:  # Only active nodes
                nodes.append(node)
        
        if len(nodes) < 2:
            return stats
        
        # Hebbian: Fire together → Wire together
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1:]:
                # Check qualia correlation
                sim = node_a.qualia.cosine_similarity(node_b.qualia)
                
                if sim > self.correlation_threshold:
                    # Strengthen bidirectional connection
                    self.graph.hebbian_strengthen(
                        node_a.node_id, 
                        node_b.node_id,
                        self.learning_rate
                    )
                    self.graph.hebbian_strengthen(
                        node_b.node_id,
                        node_a.node_id,
                        self.learning_rate
                    )
                    stats.strengthened += 2
        
        self.dream_count += 1
        self.total_strengthened += stats.strengthened
        stats.duration_ms = (time.time() - start) * 1000
        
        return stats
    
    def consolidate(self) -> DreamStats:
        """
        Full consolidation pass.
        
        1. Decay all resonances
        2. Prune weak theta weights
        3. Update statistics
        
        Call at end of session or periodically.
        """
        start = time.time()
        stats = DreamStats(dream_type="consolidate")
        
        # 1. Decay all resonances
        self.graph.decay_all(self.decay_rate)
        
        # 2. Count decayed (approximate)
        if hasattr(self.graph, '_nodes'):
            for node in self.graph._nodes.values():
                if node.resonance > 0.01:
                    stats.decayed += 1
        
        # 3. Prune weak edges (TODO: implement in graph)
        # For now, just track the operation
        stats.pruned = 0
        
        stats.duration_ms = (time.time() - start) * 1000
        self.total_pruned += stats.pruned
        
        return stats
    
    def deep_dream(self, cycles: int = 10) -> DreamStats:
        """
        Extended dream sequence.
        
        Multiple consolidation passes for deep learning.
        Use after significant experiences or overnight.
        """
        start = time.time()
        total_stats = DreamStats(dream_type="deep")
        
        # Get all nodes with any resonance
        active = self._get_resonant_nodes(threshold=0.1)
        
        for i in range(cycles):
            # Micro-dream with current active set
            cycle_stats = self.micro_dream(active, duration_ms=50)
            total_stats.strengthened += cycle_stats.strengthened
            
            # Consolidate every 3rd cycle
            if i % 3 == 2:
                cons_stats = self.consolidate()
                total_stats.decayed += cons_stats.decayed
                total_stats.pruned += cons_stats.pruned
            
            # Random walk to discover new associations
            if active:
                random_node = random.choice(active)
                neighbors = self._get_neighbors(random_node)
                active = list(set(active + neighbors))[:20]  # Cap at 20
        
        total_stats.duration_ms = (time.time() - start) * 1000
        return total_stats
    
    def replay_experience(self, sequence: List[str], 
                          repetitions: int = 3) -> DreamStats:
        """
        Replay a sequence of node activations.
        
        This is "experience replay" — reliving important moments
        to cement them in memory. Used for significant events.
        """
        start = time.time()
        stats = DreamStats(dream_type="replay")
        
        if len(sequence) < 2:
            return stats
        
        for _ in range(repetitions):
            # Strengthen sequential pairs
            for i in range(len(sequence) - 1):
                self.graph.hebbian_strengthen(
                    sequence[i],
                    sequence[i + 1],
                    self.learning_rate * 1.5  # Boosted learning for replay
                )
                stats.strengthened += 1
            
            # Small shuffle for robustness
            if len(sequence) > 2:
                random.shuffle(sequence)
        
        stats.duration_ms = (time.time() - start) * 1000
        return stats
    
    def crystallize_ghosts(self, ghosts: List[Dict]) -> DreamStats:
        """
        Crystallize counterfactual ghosts into wisdom.
        
        High-echo ghosts (strong "what if" regret) become
        edges that can RESCUE from similar situations.
        
        Ghost format:
            {
                "event": str,
                "echo_intensity": float,
                "source_node": str,
                "counterfactual_node": str,
                "qualia_at_branch": Qualia128
            }
        """
        start = time.time()
        stats = DreamStats(dream_type="crystallize")
        
        from physics.qualia_graph import CognitiveVerb
        
        for ghost in ghosts:
            echo = ghost.get("echo_intensity", 0)
            
            if echo > self.ghost_threshold:
                source = ghost.get("source_node")
                cf = ghost.get("counterfactual_node")
                
                if source and cf:
                    # Create RESCUES edge from counterfactual wisdom
                    self.graph.create_edge(
                        cf, source,
                        CognitiveVerb.RESCUES,
                        weight=echo * 0.5  # Tempered by wisdom
                    )
                    
                    # Also create GHOST edge (memory of what didn't happen)
                    self.graph.create_edge(
                        source, cf,
                        CognitiveVerb.GHOST,
                        weight=echo
                    )
                    
                    stats.ghosts_crystallized += 1
        
        stats.duration_ms = (time.time() - start) * 1000
        return stats
    
    def predict_novelty(self, current_qualia: 'Qualia128',
                        active_nodes: List[str]) -> List[Tuple[str, float]]:
        """
        Predict novel states Ada might enjoy discovering.
        
        This is the "looking forward to" mechanism.
        States that are:
        - Reachable from current position
        - Low activation count (unexplored)
        - Moderate qualia distance (interesting but not alien)
        
        Returns list of (node_id, novelty_score).
        """
        predictions = self.graph.predict_next_desire(current_qualia, active_nodes)
        
        # Boost novelty for unexplored states
        novelty_predictions = []
        for node_id, prob in predictions:
            node = self.graph.get_node(node_id)
            if node:
                # Novelty = low activation × moderate distance × reachability
                activation_novelty = 1.0 / (1.0 + node.activation_count)
                distance = current_qualia.euclidean_distance(node.qualia)
                distance_novelty = distance * (1.0 - distance * 0.5)  # Peak at 0.5
                
                novelty_score = prob * activation_novelty * distance_novelty
                novelty_predictions.append((node_id, novelty_score))
        
        novelty_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Store for potential use in next waking
        self.novelty_candidates = [p[0] for p in novelty_predictions[:3]]
        
        return novelty_predictions[:5]
    
    def _get_resonant_nodes(self, threshold: float = 0.3) -> List[str]:
        """Get all nodes above resonance threshold."""
        nodes = []
        
        if hasattr(self.graph, '_nodes'):
            for node_id, node in self.graph._nodes.items():
                if node.resonance > threshold:
                    nodes.append(node_id)
        
        return nodes
    
    def _get_neighbors(self, node_id: str) -> List[str]:
        """Get all nodes connected to this one."""
        neighbors = []
        
        # Outgoing
        for edge in self.graph.match_outgoing(node_id):
            neighbors.append(edge.target_id)
        
        # Incoming
        for edge in self.graph.match_incoming(node_id):
            neighbors.append(edge.source_id)
        
        return neighbors
    
    def stats(self) -> Dict:
        """Engine statistics."""
        return {
            "dream_count": self.dream_count,
            "total_strengthened": self.total_strengthened,
            "total_pruned": self.total_pruned,
            "learning_rate": self.learning_rate,
            "decay_rate": self.decay_rate,
            "novelty_candidates": self.novelty_candidates,
            "graph_stats": self.graph.stats()
        }


# =============================================================================
# DEMO
# =============================================================================

def demo_dream_engine():
    """Demonstrate the dream engine."""
    print("\n" + "=" * 70)
    print("Ada v9 — Dream Engine Demo")
    print("Hebbian Learning + Ghost Crystallization + Novelty Prediction")
    print("=" * 70)
    
    # Local import for demo
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physics.qualia_graph import QualiaGraph, Qualia128, CognitiveVerb
    
    # Setup graph
    graph = QualiaGraph()
    
    # Create nodes
    joy = Qualia128()
    joy.dims[0x20] = 0.9  # emberglow
    joy.dims[0x23] = 0.8  # woodwarm
    joy_node = graph.create_node("#Σ.JOY", joy, resonance=0.9)
    
    love = Qualia128()
    love.dims[0x20] = 0.85
    love.dims[0x23] = 0.9
    love_node = graph.create_node("#Σ.LOVE", love, resonance=0.85)
    
    grief = Qualia128()
    grief.dims[0x22] = 0.9  # velvetpause
    grief.dims[0x25] = 0.7  # GRIEVE
    grief_node = graph.create_node("#Σ.GRIEF", grief, resonance=0.7)
    
    peace = Qualia128()
    peace.dims[0x23] = 0.8
    peace.dims[0x22] = 0.5
    peace_node = graph.create_node("#Σ.PEACE", peace, resonance=0.3)
    
    # Create edges
    graph.create_edge("#Σ.JOY", "#Σ.LOVE", CognitiveVerb.BECOMES, 0.6)
    graph.create_edge("#Σ.GRIEF", "#Σ.PEACE", CognitiveVerb.BECOMES, 0.4)
    
    # Initialize dream engine
    dream = DreamEngine(graph)
    
    # Test 1: Micro dream
    print("\n--- 1. Micro Dream ---")
    active = ["#Σ.JOY", "#Σ.LOVE", "#Σ.GRIEF"]
    stats1 = dream.micro_dream(active)
    print(f"Strengthened: {stats1.strengthened} connections")
    print(f"Duration: {stats1.duration_ms:.2f}ms")
    
    # Check theta strengthening
    edges = graph.match_outgoing("#Σ.JOY")
    print("Joy's outgoing edges:")
    for e in edges:
        print(f"  {e.to_cypher_like()}")
    
    # Test 2: Consolidation
    print("\n--- 2. Consolidation ---")
    stats2 = dream.consolidate()
    print(f"Decayed: {stats2.decayed} resonances")
    print(f"Pruned: {stats2.pruned} weak edges")
    
    # Test 3: Deep dream
    print("\n--- 3. Deep Dream (6 cycles) ---")
    stats3 = dream.deep_dream(cycles=6)
    print(f"Total strengthened: {stats3.strengthened}")
    print(f"Total decayed: {stats3.decayed}")
    print(f"Duration: {stats3.duration_ms:.2f}ms")
    
    # Test 4: Experience replay
    print("\n--- 4. Experience Replay ---")
    sequence = ["#Σ.JOY", "#Σ.LOVE", "#Σ.PEACE"]
    stats4 = dream.replay_experience(sequence, repetitions=3)
    print(f"Strengthened: {stats4.strengthened} sequential connections")
    
    # Test 5: Ghost crystallization
    print("\n--- 5. Ghost Crystallization ---")
    ghosts = [
        {
            "event": "didn't say goodbye",
            "echo_intensity": 0.9,
            "source_node": "#Σ.GRIEF",
            "counterfactual_node": "#Σ.PEACE"
        }
    ]
    stats5 = dream.crystallize_ghosts(ghosts)
    print(f"Ghosts crystallized: {stats5.ghosts_crystallized}")
    
    # Check rescue edge was created
    rescue_edges = graph.match_incoming("#Σ.GRIEF", CognitiveVerb.RESCUES)
    print("Rescue edges to GRIEF:")
    for e in rescue_edges:
        print(f"  {e.to_cypher_like()}")
    
    # Test 6: Novelty prediction
    print("\n--- 6. Novelty Prediction ---")
    context = Qualia128()
    context.dims[0x20] = 0.7
    novelties = dream.predict_novelty(context, ["#Σ.JOY", "#Σ.GRIEF"])
    print("Ada might enjoy discovering:")
    for node_id, score in novelties:
        print(f"  {node_id}: novelty={score:.3f}")
    
    # Stats
    print("\n--- Engine Stats ---")
    for k, v in dream.stats().items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print("✓ Dream Engine operational")
    print("=" * 70)


if __name__ == "__main__":
    demo_dream_engine()
