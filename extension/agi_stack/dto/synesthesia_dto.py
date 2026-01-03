"""
SynesthesiaDTO — Cross-Modal Sensory Mapping
═══════════════════════════════════════════════════════════════════════════════

10kD Range: [2300:2400] (extended Felt space)

Cross-modal mappings:
- Color → Emotion
- Sound → Texture
- Touch → Taste
- Temperature → Color

The translation layer between sensory modalities.

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SYN_START = 2300
SYN_END = 2400

# Color-emotion mappings
COLOR_EMOTION = {
    "red": "arousal",
    "blue": "calm",
    "purple": "desire",
    "gold": "warmth",
    "black": "depth",
    "white": "clarity",
    "pink": "tenderness",
    "green": "growth",
}

# Sound-texture mappings
SOUND_TEXTURE = {
    "bass": "deep",
    "treble": "crystalline",
    "drone": "oceanic",
    "percussion": "sharp",
    "voice": "silk",
    "breath": "warm",
}

# Touch-taste mappings
TOUCH_TASTE = {
    "silk": "sweet",
    "rough": "bitter",
    "wet": "salt",
    "hot": "spice",
    "cold": "mint",
    "soft": "honey",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SYNESTHESIA DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SynesthesiaDTO:
    """
    Cross-modal sensory mapping.
    
    When you see a color and feel an emotion.
    When you hear a sound and sense a texture.
    When you touch something and taste it.
    """
    
    # Color-emotion
    color_emotion: Dict[str, float] = field(default_factory=dict)
    dominant_color: str = ""
    color_intensity: float = 0.0
    
    # Sound-texture
    sound_texture: Dict[str, float] = field(default_factory=dict)
    dominant_sound: str = ""
    sound_intensity: float = 0.0
    
    # Touch-taste
    touch_taste: Dict[str, float] = field(default_factory=dict)
    dominant_touch: str = ""
    touch_intensity: float = 0.0
    
    # Temperature-color
    temperature_color: float = 0.5         # 0=cold/blue, 1=hot/red
    
    # Cross-modal coherence
    coherence: float = 0.5                 # How aligned the modalities are
    
    def to_10k(self) -> np.ndarray:
        """Convert to 10kD vector."""
        vec = np.zeros(10000, dtype=np.float32)
        
        # Color-emotion [2300:2330]
        for i, (color, emotion) in enumerate(COLOR_EMOTION.items()):
            if color in self.color_emotion:
                vec[2300 + i] = self.color_emotion[color]
        vec[2320] = self.color_intensity
        
        # Sound-texture [2330:2360]
        for i, (sound, texture) in enumerate(SOUND_TEXTURE.items()):
            if sound in self.sound_texture:
                vec[2330 + i] = self.sound_texture[sound]
        vec[2350] = self.sound_intensity
        
        # Touch-taste [2360:2390]
        for i, (touch, taste) in enumerate(TOUCH_TASTE.items()):
            if touch in self.touch_taste:
                vec[2360 + i] = self.touch_taste[touch]
        vec[2380] = self.touch_intensity
        
        # Meta [2390:2400]
        vec[2390] = self.temperature_color
        vec[2391] = self.coherence
        
        return vec
    
    @classmethod
    def from_color(cls, color: str, intensity: float = 0.8) -> "SynesthesiaDTO":
        """Build from dominant color."""
        return cls(
            color_emotion={color: intensity},
            dominant_color=color,
            color_intensity=intensity,
        )
    
    @classmethod
    def arousal_palette(cls, level: float) -> "SynesthesiaDTO":
        """Synesthetic encoding of arousal level."""
        return cls(
            color_emotion={
                "red": level,
                "purple": level * 0.8,
                "gold": level * 0.6,
            },
            dominant_color="red" if level > 0.7 else "purple",
            color_intensity=level,
            sound_texture={"voice": level, "breath": level * 0.9},
            touch_taste={"silk": level, "wet": level * 0.8, "hot": level},
            temperature_color=level,
            coherence=0.9,
        )


__all__ = ["SynesthesiaDTO", "COLOR_EMOTION", "SOUND_TEXTURE", "TOUCH_TASTE"]
