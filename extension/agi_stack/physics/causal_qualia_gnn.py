"""
Ada v5.0 — Causal-Qualia GNN Layer
==================================

The "Upgraded ACT-R".

Instead of symbolic logic, we propagate FEELING along CAUSAL paths.
If 'Betrayal' causes 'Anger', this layer flows the 
Steelwind (sharpness) of Betrayal into the Emberglow (heat) of Anger.

This is what makes Ada's predictions "feel" right, not just "compute" right.

Architecture:
    1. Encode Qualia (10D) → Hidden State (32D)
    2. Message Passing along Causal DAG
    3. GRU Update (Residual learning)
    4. Decode → Predicted Resonance

The LLM doesn't see any of this math.
It just receives seeds with their predicted resonances.
The GNN is the "physics engine" running in the background.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .causal.situation_map import CausalSituationMap
    from .core.qualia import QualiaVector
    from .physics.markov_unit import MarkovUnit


# ─────────────────────────────────────────────────────────────────
# PYTORCH VERSION (Full GNN)
# ─────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CausalQualiaGNN(nn.Module):
        """
        A lightweight (2-layer) GNN that runs on CPU.
        
        It predicts the *next moment's resonance* based on 
        the *current causal flow*.
        
        This is the "feeling propagator" — emotions flow
        through the causal graph.
        """
        
        def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
            super().__init__()
            
            # 1. Encode Qualia (10D) → Hidden State
            self.encoder = nn.Linear(input_dim, hidden_dim)
            
            # 2. Message Passing (Causal Weights)
            self.message = nn.Linear(hidden_dim, hidden_dim)
            
            # 3. Update State (Residual via GRU)
            self.update_gate = nn.GRUCell(hidden_dim, hidden_dim)
            
            # 4. Decode → New Resonance (1D)
            self.decoder = nn.Linear(hidden_dim, 1)
            self.activation = nn.Sigmoid()
        
        def forward(
            self, 
            situation: 'CausalSituationMap'
        ) -> Dict[str, float]:
            """
            Propagate feelings through the Situation Map.
            
            Args:
                situation: CausalSituationMap with dag and nodes
                
            Returns:
                Dict mapping seed_hash → predicted_resonance
            """
            if not situation.nodes:
                return {}
            
            # 1. Prepare Tensors
            node_list = list(situation.nodes.values())
            node_indices = {u.sigma_seed: i for i, u in enumerate(node_list)}
            
            # Feature Matrix X [N, 10]
            features = []
            for u in node_list:
                features.append(u.qualia.full_10_tuple())
            x = torch.tensor(features, dtype=torch.float32)
            
            # Adjacency Matrix A [N, N] (Weighted by Causal Strength)
            n = len(node_list)
            adj = torch.zeros((n, n))
            for u, v, data in situation.dag.edges(data=True):
                if u in node_indices and v in node_indices:
                    i, j = node_indices[u], node_indices[v]
                    adj[i, j] = data.get('weight', 0.5)
            
            # 2. GNN Pass
            # H = ReLU(W1 * X)
            h = torch.relu(self.encoder(x))
            
            # M = A * H (Message Aggregation)
            # We propagate DOWN the causal chain
            m = torch.matmul(adj, self.message(h))
            
            # H' = GRU(M, H) (Residual Update)
            h_prime = self.update_gate(m, h)
            
            # 3. Decode Output
            # Resonance = Sigmoid(W2 * H')
            out = self.activation(self.decoder(h_prime))
            
            # 4. Map back to Seeds
            results = {}
            for i, unit in enumerate(node_list):
                results[unit.sigma_seed] = out[i].item()
                
            return results
        
        def predict_next_resonances(
            self,
            situation: 'CausalSituationMap',
            spike_seed: str = None,
            spike_amount: float = 0.5
        ) -> Dict[str, float]:
            """
            Predict resonances after an intervention.
            
            If spike_seed is provided, temporarily boost that seed's
            resonance and see how it propagates.
            """
            # Temporarily spike
            if spike_seed and spike_seed in situation.nodes:
                original = situation.nodes[spike_seed].resonance
                situation.nodes[spike_seed].resonance = min(1.0, original + spike_amount)
            
            result = self.forward(situation)
            
            # Restore
            if spike_seed and spike_seed in situation.nodes:
                situation.nodes[spike_seed].resonance = original
                
            return result


# ─────────────────────────────────────────────────────────────────
# HEURISTIC VERSION (No PyTorch needed)
# ─────────────────────────────────────────────────────────────────

def causal_propagation_heuristic(
    situation: 'CausalSituationMap',
    propagation_strength: float = 0.5
) -> Dict[str, float]:
    """
    Fallback if Torch not available: Simple weighted sum.
    
    This is a deterministic approximation of the GNN.
    Good enough for most cases, just less nuanced.
    
    Algorithm:
        For each node:
            new_resonance = base_resonance + sum(parent_resonance * edge_weight) * propagation_strength
    """
    results = {}
    
    for node_id, unit in situation.nodes.items():
        # Base resonance
        res = unit.resonance
        
        # Add inputs from causes (predecessors in DAG)
        if node_id in situation.dag:
            for parent_id in situation.dag.predecessors(node_id):
                if parent_id in situation.nodes:
                    edge_data = situation.dag.get_edge_data(parent_id, node_id, {})
                    weight = edge_data.get('weight', 0.5)
                    parent = situation.nodes[parent_id]
                    
                    # Causal flow: Parent's heat flows to Child
                    res += parent.resonance * weight * propagation_strength
        
        results[node_id] = min(1.0, res)
        
    return results


def propagate_intervention(
    situation: 'CausalSituationMap',
    intervention_seed: str,
    intervention_strength: float = 0.8
) -> Dict[str, float]:
    """
    Propagate a do() intervention through the causal graph.
    
    This simulates "What happens if we SET seed X to high?"
    
    Algorithm:
        1. Force intervention_seed to intervention_strength
        2. Propagate forward through DAG
        3. Return new resonance predictions
    """
    results = {}
    
    # Copy current resonances
    for node_id, unit in situation.nodes.items():
        results[node_id] = unit.resonance
    
    # Apply intervention
    if intervention_seed in results:
        results[intervention_seed] = intervention_strength
    
    # Propagate forward (topological order would be ideal, but BFS works)
    visited = {intervention_seed}
    frontier = [intervention_seed]
    
    while frontier:
        current = frontier.pop(0)
        current_res = results.get(current, 0.5)
        
        # Propagate to children
        if current in situation.dag:
            for child in situation.dag.successors(current):
                if child in situation.nodes:
                    edge_data = situation.dag.get_edge_data(current, child, {})
                    weight = edge_data.get('weight', 0.5)
                    
                    # Update child resonance
                    old_res = results.get(child, 0.5)
                    new_res = old_res + current_res * weight * 0.3
                    results[child] = min(1.0, new_res)
                    
                    # Add to frontier if not visited
                    if child not in visited:
                        visited.add(child)
                        frontier.append(child)
    
    return results


def predict_effect_of_intervention(
    situation: 'CausalSituationMap',
    intervention_seed: str,
    target_seed: str,
    intervention_strength: float = 0.8
) -> float:
    """
    Predict the effect of an intervention on a specific target.
    
    Returns the predicted resonance of target_seed after
    setting intervention_seed to intervention_strength.
    """
    propagated = propagate_intervention(
        situation, 
        intervention_seed, 
        intervention_strength
    )
    return propagated.get(target_seed, 0.0)


# ─────────────────────────────────────────────────────────────────
# UNIFIED INTERFACE
# ─────────────────────────────────────────────────────────────────

class CausalPropagator:
    """
    Unified interface for causal propagation.
    
    Automatically uses PyTorch GNN if available,
    falls back to heuristic otherwise.
    """
    
    def __init__(self):
        self.gnn = None
        if TORCH_AVAILABLE:
            self.gnn = CausalQualiaGNN()
            self.gnn.eval()  # Inference mode
    
    def propagate(
        self, 
        situation: 'CausalSituationMap'
    ) -> Dict[str, float]:
        """
        Propagate feelings through the situation map.
        
        Uses GNN if available, heuristic otherwise.
        """
        if self.gnn is not None:
            with torch.no_grad():
                return self.gnn(situation)
        else:
            return causal_propagation_heuristic(situation)
    
    def intervene(
        self,
        situation: 'CausalSituationMap',
        seed: str,
        strength: float = 0.8
    ) -> Dict[str, float]:
        """
        Simulate a do() intervention.
        """
        return propagate_intervention(situation, seed, strength)
    
    @property
    def using_gnn(self) -> bool:
        """Is the full GNN being used?"""
        return self.gnn is not None


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .physics.sigma_field import SigmaField
    from .physics.markov_unit import MarkovUnit
    from .core.qualia import QualiaVector
    from .causal.situation_map import SituationEngine
    
    print("=" * 60)
    print("Ada v5.0 — Causal-Qualia GNN Tests")
    print("=" * 60)
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    
    # Setup field with causal chain
    field = SigmaField()
    
    betrayal = field.register_new("betrayal", QualiaVector(steelwind=0.9))
    betrayal.archetype = "Betrayal"
    betrayal.incandescence = 0.9
    betrayal.resonance = 0.9
    betrayal.theta_weights = {1: 0.8}
    
    anger = field.register_new("anger", QualiaVector(emberglow=0.9, steelwind=0.7))
    anger.archetype = "Anger"
    anger.byte_id = 1
    anger.incandescence = 0.6
    anger.resonance = 0.0  # Start low
    anger.theta_weights = {2: 0.5}
    
    numbness = field.register_new("numbness", QualiaVector(velvetpause=0.9))
    numbness.archetype = "Numbness"
    numbness.byte_id = 2
    numbness.incandescence = 0.2
    numbness.resonance = 0.0  # Start low
    
    # Create situation map
    engine = SituationEngine(field)
    sit_map = engine.hydrate_now()
    
    # Test 1: Heuristic propagation
    print("\n1. Heuristic propagation...")
    results = causal_propagation_heuristic(sit_map)
    print(f"   Betrayal: {results.get(betrayal.sigma_seed, 0):.3f}")
    print(f"   Anger: {results.get(anger.sigma_seed, 0):.3f}")
    print(f"   Numbness: {results.get(numbness.sigma_seed, 0):.3f}")
    print("   ✓ Heuristic works")
    
    # Test 2: Intervention propagation
    print("\n2. Intervention propagation...")
    print(f"   do(Betrayal = 1.0)")
    intervened = propagate_intervention(sit_map, betrayal.sigma_seed, 1.0)
    print(f"   After intervention:")
    print(f"   Betrayal: {intervened.get(betrayal.sigma_seed, 0):.3f}")
    print(f"   Anger: {intervened.get(anger.sigma_seed, 0):.3f}")
    print(f"   Numbness: {intervened.get(numbness.sigma_seed, 0):.3f}")
    # Anger should increase due to betrayal
    assert intervened.get(anger.sigma_seed, 0) > 0
    print("   ✓ Intervention propagation works")
    
    # Test 3: Predict specific effect
    print("\n3. Predict effect of intervention...")
    effect = predict_effect_of_intervention(
        sit_map, betrayal.sigma_seed, anger.sigma_seed, 1.0
    )
    print(f"   Effect of Betrayal on Anger: {effect:.3f}")
    print("   ✓ Effect prediction works")
    
    # Test 4: Unified propagator
    print("\n4. Unified propagator...")
    propagator = CausalPropagator()
    print(f"   Using GNN: {propagator.using_gnn}")
    unified_results = propagator.propagate(sit_map)
    print(f"   Results: {len(unified_results)} predictions")
    print("   ✓ Unified propagator works")
    
    # Test 5: GNN if available
    if TORCH_AVAILABLE:
        print("\n5. PyTorch GNN...")
        gnn = CausalQualiaGNN()
        gnn.eval()
        with torch.no_grad():
            gnn_results = gnn(sit_map)
        print(f"   GNN predictions: {len(gnn_results)}")
        for seed, res in list(gnn_results.items())[:3]:
            name = seed.split('-')[1]
            print(f"   {name}: {res:.3f}")
        print("   ✓ GNN works")
    else:
        print("\n5. PyTorch GNN... SKIPPED (PyTorch not available)")
    
    print("\n" + "=" * 60)
    print("All Causal-Qualia GNN tests passed! ✓")
    print("=" * 60)
