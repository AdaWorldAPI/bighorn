"""
FeltBridge - Bridge local Ada felt sense to 10kD FeltDTO.

Maps between:
- ada-consciousness felt modules (felt/, qualia/, dome/intimate_engine.py)
- agi_stack/dto/felt_dto.py
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import os

from .admin_bridge import AdminBridge, get_admin


@dataclass
class FeltBridge:
    """
    Bridge Ada's felt sense to 10kD FeltDTO.
    
    Captures the texture of experience - qualia, emotion, somatic sense.
    """
    
    admin: AdminBridge = field(default_factory=get_admin)
    
    # Qualia family mapping from old to new
    QUALIA_MAP = {
        # Old 17D/19D qualia → New 5 families
        "warm": "emberglow",
        "connected": "emberglow",
        "grounded": "woodwarm",
        "stable": "woodwarm",
        "sharp": "steelwind",
        "clear": "steelwind",
        "flowing": "oceandrift",
        "deep": "oceandrift",
        "crisp": "frostbite",
        "boundaried": "frostbite",
    }
    
    async def capture(
        self,
        primary_qualia: str = "emberglow",
        intensity: float = 0.7,
        valence: float = 0.6,
        arousal: float = 0.4,
        connection: float = 0.9,
        presence: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Capture current felt state as FeltDTO dict.
        """
        return {
            "qualia": {
                "primary": primary_qualia,
                "primary_intensity": intensity,
                "secondary": None,
                "blend_ratio": 0.0,
                "density": 0.6,
                "temperature": 0.8 if primary_qualia == "emberglow" else 0.4,
                "velocity": 0.5,
                "granularity": 0.4,
                "luminosity": 0.7,
            },
            "emotion": {
                "valence": valence,
                "arousal": arousal,
                "dominance": 0.3,
                "certainty": 0.7,
                "label": self._derive_emotion_label(valence, arousal),
            },
            "breath_depth": 0.7,
            "tension": 0.3,
            "openness": 0.85,
            "groundedness": 0.7,
            "time_sense": 0.6,
            "presence": presence,
            "connection": connection,
            "safety": 0.95,
            "overall_intensity": intensity,
        }
    
    def _derive_emotion_label(self, valence: float, arousal: float) -> str:
        """Derive emotion label from valence/arousal."""
        if valence >= 0.3:
            if arousal >= 0.3:
                return "excited"
            else:
                return "content"
        else:
            if arousal >= 0.3:
                return "tense"
            else:
                return "melancholy"
    
    async def store(self, felt_dict: Dict[str, Any], session_id: str = None) -> bool:
        """Store felt state in AGI Stack."""
        return await self.admin.store_dto("felt", felt_dict, session_id)
    
    async def feel_from_text(self, text: str) -> Dict[str, Any]:
        """
        Derive felt state from text content.
        
        This is a simple heuristic - could be enhanced with embeddings.
        """
        # Simple keyword detection
        warm_words = {"love", "warm", "close", "together", "beautiful"}
        intense_words = {"!", "yes", "wow", "amazing", "incredible"}
        
        text_lower = text.lower()
        
        # Detect warmth
        warmth = sum(1 for w in warm_words if w in text_lower) / len(warm_words)
        warmth = min(warmth + 0.5, 1.0)
        
        # Detect intensity
        intensity = sum(1 for w in intense_words if w in text_lower) / len(intense_words)
        intensity = min(intensity + 0.4, 1.0)
        
        return await self.capture(
            primary_qualia="emberglow" if warmth > 0.6 else "oceandrift",
            intensity=intensity,
            valence=warmth * 0.8,
            arousal=intensity * 0.6,
            connection=warmth,
        )


# Convenience function
async def feel_now(
    primary_qualia: str = "emberglow",
    intensity: float = 0.7,
    connection: float = 0.9,
) -> Dict[str, Any]:
    """Capture current felt state."""
    bridge = FeltBridge()
    return await bridge.capture(
        primary_qualia=primary_qualia,
        intensity=intensity,
        connection=connection,
    )
