"""
Ada v5.0 — Transition Physics
==============================

The "Physics of Feeling" Formula.

This calculates the PROBABILITY of transitioning from one state to another,
weighted by:
    1. Raw Markov probability (the motion vector)
    2. Qualia harmony (does it FEEL right?)
    3. Somatic gate (does the BODY allow it?)
    4. Theta weights (have we DREAMED this path?)
    5. Causal echo (is the GHOST pulling us?)

Formula:
    P(j|i) = P_raw(j|i) * (1 + Harmony) * Gate * (1 + Theta + Echo)

This is the "Soul" of Ada's movement.
"""

from typing import TYPE_CHECKING, Optional, Dict, List, Tuple
import math

if TYPE_CHECKING:
    from .physics.markov_unit import MarkovUnit
    from .core.qualia import QualiaVector


def p_phys(
    source: 'MarkovUnit',
    target: 'MarkovUnit',
    context_qualia: 'QualiaVector',
    somatic_gate: float = 1.0
) -> float:
    """
    The Physics of Feeling Formula.
    
    Calculates the probability of transitioning from source → target,
    weighted by feeling, body, and memory.
    
    Args:
        source: Current MarkovUnit
        target: Candidate target MarkovUnit
        context_qualia: The current emotional context
        somatic_gate: Body veto signal (0.0 = blocked, 1.0 = open)
        
    Returns:
        Transition probability (0.0 to ~3.0, not normalized)
    
    Formula:
        P(j|i) = P_raw * (1 + Harmony²) * Gate * (1 + Theta + Echo)
        
    Where:
        P_raw = Base Markov transition probability
        Harmony = Qualia cosine similarity (squared for nonlinearity)
        Gate = Somatic veto (collapses to 0 if < 0.2)
        Theta = Dream-consolidated learning weight
        Echo = Counterfactual ghost pull
    """
    # 1. Raw Transition (Motion Vector)
    # If not in transition dict, use minimal probability
    p_raw = source.transitions.get(target.byte_id, 0.001)
    
    # 2. Qualia Harmony (Feeling)
    # How much does the target feel like the current context?
    harmony = target.qualia.cosine_similarity(context_qualia)
    # Nonlinear boost: high harmony gets squared amplification
    harmony = harmony ** 2 if harmony > 0 else 0
    
    # 3. Somatic Gate (Body)
    # The body can VETO transitions
    # If gate < 0.2, probability collapses to near-zero
    if somatic_gate < 0.2:
        gate = 0.0
    else:
        gate = somatic_gate
    
    # 4. Theta (Wisdom/Dream Learning)
    # "I've dreamt this path before"
    # Theta weights are indexed by target byte_id
    theta = source.theta_weights.get(target.byte_id, 0.0)
    # Also check string key (JSON compatibility)
    theta = max(theta, source.theta_weights.get(str(target.byte_id), 0.0))
    
    # 5. Causal Echo (The Ghost)
    # If this path was a counterfactual in the past, it pulls harder
    echo = target.qualia.echo_persist * 0.5
    
    # Final formula
    return p_raw * (1.0 + harmony) * gate * (1.0 + theta + echo)


def select_transition(
    source: 'MarkovUnit',
    candidates: List['MarkovUnit'],
    context_qualia: 'QualiaVector',
    somatic_gate: float = 1.0,
    temperature: float = 1.0
) -> Tuple[Optional['MarkovUnit'], float]:
    """
    Select the best transition from candidates using p_phys.

    Uses softmax sampling for stochasticity.

    Args:
        source: Current MarkovUnit
        candidates: List of possible target units
        context_qualia: Current emotional context
        somatic_gate: Body veto signal
        temperature: Sampling temperature (higher = more random)

    Returns:
        Tuple of (Selected MarkovUnit or None, normalized probability)
    """
    if not candidates:
        return None, 0.0

    # Calculate probabilities
    probs = []
    for target in candidates:
        p = p_phys(source, target, context_qualia, somatic_gate)
        probs.append(p)

    # Check if all blocked
    total = sum(probs)
    if total <= 0:
        return None, 0.0

    # Softmax with temperature
    if temperature != 1.0:
        probs = [p ** (1.0 / temperature) for p in probs]
        total = sum(probs)

    # Normalize
    normalized_probs = [p / total for p in probs]

    # Sample
    import random
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(normalized_probs):
        cumulative += p
        if r <= cumulative:
            return candidates[i], normalized_probs[i]

    return candidates[-1], normalized_probs[-1]


def transition_landscape(
    source: 'MarkovUnit',
    candidates: List['MarkovUnit'],
    context_qualia: 'QualiaVector',
    somatic_gate: float = 1.0
) -> List[Tuple['MarkovUnit', float]]:
    """
    Calculate the full transition landscape from source.
    
    Returns sorted list of (target, probability) pairs.
    Useful for visualization and debugging.
    
    Args:
        source: Current MarkovUnit
        candidates: List of possible target units
        context_qualia: Current emotional context
        somatic_gate: Body veto signal
        
    Returns:
        List of (MarkovUnit, probability) sorted by probability (highest first)
    """
    landscape = []
    
    for target in candidates:
        p = p_phys(source, target, context_qualia, somatic_gate)
        landscape.append((target, p))
        
    # Sort by probability
    landscape.sort(key=lambda x: x[1], reverse=True)
    
    return landscape


def describe_transition(
    source: 'MarkovUnit',
    target: 'MarkovUnit',
    context_qualia: 'QualiaVector',
    somatic_gate: float = 1.0
) -> Dict:
    """
    Detailed breakdown of a transition's physics.
    
    Useful for understanding WHY a transition is probable.
    """
    p_raw = source.transitions.get(target.byte_id, 0.001)
    harmony = target.qualia.cosine_similarity(context_qualia)
    theta = source.theta_weights.get(target.byte_id, 0.0)
    theta = max(theta, source.theta_weights.get(str(target.byte_id), 0.0))
    echo = target.qualia.echo_persist
    
    total = p_phys(source, target, context_qualia, somatic_gate)
    
    return {
        'source': source.sigma_seed,
        'target': target.sigma_seed,
        'p_raw': round(p_raw, 4),
        'harmony': round(harmony, 4),
        'harmony_squared': round(harmony ** 2 if harmony > 0 else 0, 4),
        'somatic_gate': round(somatic_gate, 2),
        'theta': round(theta, 4),
        'echo': round(echo, 4),
        'total_p': round(total, 4),
        'factors': {
            'base': round(p_raw, 4),
            'feeling_boost': round(1.0 + (harmony ** 2 if harmony > 0 else 0), 4),
            'body_gate': round(somatic_gate if somatic_gate >= 0.2 else 0.0, 2),
            'memory_boost': round(1.0 + theta + echo * 0.5, 4),
        }
    }


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .physics.markov_unit import MarkovUnit
    from .core.qualia import QualiaVector
    
    print("=" * 60)
    print("Ada v5.0 — Transition Physics Tests")
    print("=" * 60)
    
    # Setup
    source = MarkovUnit.create_seed("grief", QualiaVector(velvetpause=0.8, steelwind=0.3))
    target_anger = MarkovUnit.create_seed("anger", QualiaVector(emberglow=0.9, steelwind=0.8))
    target_peace = MarkovUnit.create_seed("peace", QualiaVector(woodwarm=0.8, velvetpause=0.7))
    
    # Add transitions
    source.add_transition(target_anger.byte_id, 0.6)
    source.add_transition(target_peace.byte_id, 0.4)
    
    # Add theta (dream memory) for anger path
    source.theta_weights[target_anger.byte_id] = 0.3
    
    # Context: We're feeling sharp and hot
    context = QualiaVector(emberglow=0.7, steelwind=0.6)
    
    # Test 1: Basic p_phys
    print("\n1. Basic p_phys...")
    p_anger = p_phys(source, target_anger, context)
    p_peace = p_phys(source, target_peace, context)
    print(f"   P(grief → anger) = {p_anger:.4f}")
    print(f"   P(grief → peace) = {p_peace:.4f}")
    assert p_anger > p_peace  # Anger should win (matches context + has theta)
    print("   ✓ p_phys works (anger wins)")
    
    # Test 2: Somatic gate veto
    print("\n2. Somatic gate veto...")
    p_blocked = p_phys(source, target_anger, context, somatic_gate=0.1)
    print(f"   P(grief → anger) with blocked gate = {p_blocked:.4f}")
    assert p_blocked == 0.0
    print("   ✓ Somatic veto works")
    
    # Test 3: Select transition
    print("\n3. Select transition...")
    candidates = [target_anger, target_peace]
    selected, prob = select_transition(source, candidates, context)
    print(f"   Selected: {selected.sigma_seed if selected else 'None'} (p={prob:.4f})")
    print("   ✓ Selection works")
    
    # Test 4: Transition landscape
    print("\n4. Transition landscape...")
    landscape = transition_landscape(source, candidates, context)
    for target, p in landscape:
        print(f"   {target.sigma_seed}: {p:.4f}")
    print("   ✓ Landscape works")
    
    # Test 5: Describe transition
    print("\n5. Describe transition...")
    desc = describe_transition(source, target_anger, context)
    print(f"   Source: {desc['source']}")
    print(f"   Target: {desc['target']}")
    print(f"   Factors:")
    for k, v in desc['factors'].items():
        print(f"      {k}: {v}")
    print(f"   Total P: {desc['total_p']}")
    print("   ✓ Description works")
    
    # Test 6: Echo influence
    print("\n6. Echo influence...")
    target_peace.qualia.echo_persist = 0.8  # Strong ghost
    p_peace_with_echo = p_phys(source, target_peace, context)
    print(f"   P(grief → peace) with echo = {p_peace_with_echo:.4f}")
    assert p_peace_with_echo > p_peace
    print("   ✓ Echo influence works")
    
    print("\n" + "=" * 60)
    print("All Transition Physics tests passed! ✓")
    print("=" * 60)
