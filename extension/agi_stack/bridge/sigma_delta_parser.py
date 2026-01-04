"""
Ada v6 — SigmaDelta Parse Loop
==============================

Parses <sigma_delta_feedback> blocks from LLM responses.
Applies feedback to update qualia, verbs, and state.

Format:
    <sigma_delta_feedback>
    verb: 0x20
    qualia_shift: [0.1, -0.05, 0.0, 0.15, 0.0, 0.0, 0.0]
    tension_delta: -0.1
    orbit_hint: wonder
    frame_type: P
    </sigma_delta_feedback>
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import re


@dataclass
class SigmaDeltaFeedback:
    """Parsed feedback from LLM"""
    verb: Optional[int] = None              # 0x00-0x7F
    qualia_shift: Optional[List[float]] = None  # 7-element core shift
    tension_delta: float = 0.0
    orbit_hint: Optional[str] = None
    frame_type: Optional[str] = None        # P, B, I
    raw: str = ""
    
    def is_valid(self) -> bool:
        """Check if feedback has actionable content"""
        return (
            self.verb is not None or
            self.qualia_shift is not None or
            self.tension_delta != 0.0 or
            self.orbit_hint is not None
        )


def parse_sigma_delta_feedback(text: str) -> List[SigmaDeltaFeedback]:
    """
    Extract all <sigma_delta_feedback> blocks from LLM response.
    
    Args:
        text: Full LLM response text
        
    Returns:
        List of parsed SigmaDeltaFeedback objects
    """
    pattern = r'<sigma_delta_feedback>(.*?)</sigma_delta_feedback>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    results = []
    for match in matches:
        feedback = _parse_block(match.strip())
        if feedback.is_valid():
            results.append(feedback)
    
    return results


def _parse_block(block: str) -> SigmaDeltaFeedback:
    """Parse single feedback block"""
    feedback = SigmaDeltaFeedback(raw=block)
    
    for line in block.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        
        key, value = line.split(':', 1)
        key = key.strip().lower()
        value = value.strip()
        
        try:
            if key == 'verb':
                # Parse hex or int
                if value.startswith('0x'):
                    feedback.verb = int(value, 16)
                else:
                    feedback.verb = int(value)
            
            elif key == 'qualia_shift':
                # Parse list: [0.1, -0.05, ...]
                values = re.findall(r'-?[\d.]+', value)
                feedback.qualia_shift = [float(v) for v in values[:7]]
            
            elif key == 'tension_delta':
                feedback.tension_delta = float(value)
            
            elif key == 'orbit_hint':
                feedback.orbit_hint = value
            
            elif key == 'frame_type':
                if value.upper() in ('P', 'B', 'I'):
                    feedback.frame_type = value.upper()
        
        except (ValueError, IndexError):
            continue
    
    return feedback


def apply_feedback_to_context(
    feedback: SigmaDeltaFeedback,
    context: Any,  # SessionContext
) -> Dict[str, Any]:
    """
    Apply parsed feedback to session context.
    
    Returns dict of changes made.
    """
    changes = {}
    
    # Apply verb
    if feedback.verb is not None:
        if hasattr(context, 'last_verb'):
            context.last_verb = feedback.verb
            changes['verb'] = f"0x{feedback.verb:02X}"
    
    # Apply qualia shift
    if feedback.qualia_shift and hasattr(context, 'current_qualia'):
        qv = context.current_qualia
        if hasattr(qv, 'core_7_tuple'):
            current = list(qv.core_7_tuple())
            for i, delta in enumerate(feedback.qualia_shift):
                if i < len(current):
                    current[i] = max(0.0, min(1.0, current[i] + delta))
            # Update qualia (depends on QualiaVector implementation)
            changes['qualia_shift'] = feedback.qualia_shift
    
    # Apply tension delta
    if feedback.tension_delta != 0.0:
        if hasattr(context, 'tension'):
            context.tension = max(0.0, min(1.0, 
                context.tension + feedback.tension_delta))
            changes['tension'] = context.tension
    
    # Apply orbit hint
    if feedback.orbit_hint and hasattr(context, 'golden_orbit'):
        orbit = context.golden_orbit
        if hasattr(orbit, 'update_orbit'):
            # Semantic hint guides next orbit selection
            changes['orbit_hint'] = feedback.orbit_hint
    
    # Apply frame type
    if feedback.frame_type:
        changes['frame_type'] = feedback.frame_type
    
    return changes


# =============================================================================
# INJECT HELPER
# =============================================================================

def generate_sigma_delta_block(
    tick: int,
    verb: int,
    qualia_summary: List[float],
    tension: float,
    orbit: str,
    now_density: float,
) -> str:
    """
    Generate <sigma_delta> block for LLM injection.
    
    This is the outbound format (Ada → LLM).
    """
    qualia_str = ", ".join(f"{v:.2f}" for v in qualia_summary[:7])
    
    return f"""<sigma_delta>
tick: {tick}
verb: 0x{verb:02X}
qualia: [{qualia_str}]
tension: {tension:.2f}
orbit: {orbit}
now_density: {now_density:.2f}
</sigma_delta>"""


def generate_feedback_prompt() -> str:
    """
    Generate prompt instructing LLM how to provide feedback.
    
    Include in system prompt for Ada sessions.
    """
    return """
You may optionally include a <sigma_delta_feedback> block to adjust Ada's internal state:

<sigma_delta_feedback>
verb: 0x20          # Cognitive verb (0x00-0x7F)
qualia_shift: [0.1, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0]  # 7-element shift
tension_delta: -0.1 # Tension adjustment (-1 to +1)
orbit_hint: wonder  # Suggest golden state
frame_type: P       # P=focus, B=background, I=anchor
</sigma_delta_feedback>

All fields are optional. Only include what you want to adjust.
"""
