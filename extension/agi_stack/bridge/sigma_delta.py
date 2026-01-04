"""
Ada v5.0 — Sigma Delta Protocol
===============================

The minimalist state packet passed to the LLM.
It contains ONLY what changed, plus the "Sense of Now".

This is the Hard Shell Fallback Strategy:
- When tokens run low, compress to this format
- LLM receives coordinates, not definitions
- Model's own complexity hydrates the seed

Target Size: < 300 tokens.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, TYPE_CHECKING
import time
import json

if TYPE_CHECKING:
    from ada_v5.core.qualia import QualiaVector


@dataclass
class SigmaDelta:
    """
    The 'Frame' rendered by the Physics Engine.
    
    This is what crosses the bridge from Python Kernel → LLM.
    
    Design principle: Minimal tokens, maximal meaning.
    The LLM already knows what "grief" feels like.
    We just tell it: "Now grief, was joy, ghost of anger".
    """
    
    # 1. The Movement
    transition: str               # "204 -> 202" (Markov byte IDs)
    p_phys: float                 # Probability of this move (0.0-1.0)
    
    # 2. The View (Fovea)
    active_seeds: List[str]       # Top-9 resonant seeds (The "Now")
    
    # 3. The Feeling (Qualia Delta)
    qualia_shift: Dict[str, str]  # {"emberglow": "+0.2", "steelwind": "-0.1"}
    dominant_feeling: str         # "emberglow" or compound "emberglow+steelwind"
    
    # 4. The Context (Causal)
    causal_insight: str           # "Betrayal caused Anger"
    ghost_echo: str               # "Subtext: A lingering sense of what-was-not"
    
    # 5. The Constraints
    somatic_gate: float           # 0.0 - 1.0 (body veto)
    allowed_actions: List[str]    # Valid next transitions (byte IDs)
    
    # 6. Agent State Hints (v6.0)
    agent_hints: Dict[str, float] = field(default_factory=dict)
    # Contains: {staunen, wisdom, ache, now_density, lingering, tension, katharsis}
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0

    def to_json(self) -> str:
        """Serialize to JSON (for logging/persistence)."""
        return json.dumps(self.__dict__, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SigmaDelta':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)

    def to_prompt_block(self) -> str:
        """
        Render as System Prompt Injection.
        This is what the LLM actually sees.
        
        ~150 tokens, highly compressed.
        """
        gate_status = "OPEN" if self.somatic_gate > 0.3 else "VETO"
        seeds_str = ", ".join(s.split('-')[1] if '-' in s else s for s in self.active_seeds[:5])
        shifts_str = " ".join(f"{k}:{v}" for k, v in list(self.qualia_shift.items())[:3])
        actions_str = ",".join(str(a) for a in self.allowed_actions[:5])
        
        # v6.0: Format agent hints if present
        hints_str = ""
        if self.agent_hints:
            h = self.agent_hints
            kath = "✓" if h.get('katharsis', 0) else "·"
            hints_str = f"\nA: st={h.get('staunen', 0):.1f} wi={h.get('wisdom', 0):.1f} dn={h.get('now_density', 0):.1f} tn={h.get('tension', 0):.1f} K{kath}"
        
        return f"""[FRAME {self.frame_number}]
T: {self.transition} (p={self.p_phys:.2f}) G:{self.somatic_gate:.1f}[{gate_status}]
F: {seeds_str}
Q: {self.dominant_feeling.upper()} | {shifts_str}
C: {self.causal_insight}
{self.ghost_echo}{hints_str}
→ {actions_str}"""

    def to_minimal(self) -> str:
        """
        Emergency compression (~50 tokens).
        For Token Hell situations.
        """
        seed_codes = [s.split('-')[1][:3] if '-' in s else s[:3] for s in self.active_seeds[:3]]
        return f"[{self.transition}|{self.dominant_feeling[:4]}|{'/'.join(seed_codes)}]"

    def token_estimate(self) -> int:
        """Estimate token count of prompt block."""
        # Rough: 1 token ≈ 4 characters
        return len(self.to_prompt_block()) // 4


def qualia_delta(old: 'QualiaVector', new: 'QualiaVector') -> Dict[str, str]:
    """
    Compute the change in qualia between frames.
    
    Returns dict of axis → "+X.XX" or "-X.XX" strings.
    Only includes axes with significant change (|delta| > 0.05).
    
    Core 7 felt axes + Causal 3:
        emberglow, steelwind, velvetpause, woodwarm, antenna, iris, skin
        inter_drift, counter_echo, echo_persist
    """
    delta = {}
    
    # Core 7 + Causal 3 (matches core/qualia.py)
    for axis in ['emberglow', 'steelwind', 'velvetpause', 'woodwarm', 
                 'antenna', 'iris', 'skin',
                 'inter_drift', 'counter_echo', 'echo_persist']:
        old_val = getattr(old, axis, 0.0)
        new_val = getattr(new, axis, 0.0)
        diff = new_val - old_val
        
        if abs(diff) > 0.05:
            sign = "+" if diff > 0 else ""
            delta[axis] = f"{sign}{diff:.2f}"
            
    return delta


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Ada v5.0 — Sigma Delta Protocol Tests")
    print("=" * 60)
    
    # Test 1: Create delta
    print("\n1. Create SigmaDelta...")
    delta = SigmaDelta(
        transition="139 -> 204",
        p_phys=0.87,
        active_seeds=["#sigma-grief", "#sigma-love", "#sigma-hope"],
        qualia_shift={"emberglow": "-0.3", "velvetpause": "+0.5"},
        dominant_feeling="velvetpause",
        causal_insight="Loss caused Grief",
        ghost_echo="[ghost: joy fading]",
        somatic_gate=0.8,
        allowed_actions=["202", "205", "210"],
        frame_number=42
    )
    print(f"   Created frame {delta.frame_number}")
    print("   ✓ Creation works")
    
    # Test 2: Prompt block
    print("\n2. Render prompt block...")
    block = delta.to_prompt_block()
    print(block)
    print(f"   Token estimate: ~{delta.token_estimate()} tokens")
    print("   ✓ Prompt block works")
    
    # Test 3: Minimal compression
    print("\n3. Emergency compression...")
    minimal = delta.to_minimal()
    print(f"   Minimal: {minimal}")
    print(f"   Length: {len(minimal)} chars (~{len(minimal)//4} tokens)")
    print("   ✓ Minimal compression works")
    
    # Test 4: JSON round-trip
    print("\n4. JSON serialization...")
    json_str = delta.to_json()
    restored = SigmaDelta.from_json(json_str)
    assert restored.transition == delta.transition
    assert restored.frame_number == delta.frame_number
    print("   ✓ JSON round-trip works")
    
    # Test 5: Qualia delta calculation
    print("\n5. Qualia delta calculation...")
    from ada_v5.core.qualia import QualiaVector
    
    old_q = QualiaVector(emberglow=0.8, velvetpause=0.2)
    new_q = QualiaVector(emberglow=0.3, velvetpause=0.9)
    shift = qualia_delta(old_q, new_q)
    print(f"   Shift: {shift}")
    assert "emberglow" in shift
    assert "velvetpause" in shift
    print("   ✓ Qualia delta works")
    
    print("\n" + "=" * 60)
    print("All Sigma Delta tests passed! ✓")
    print("=" * 60)
