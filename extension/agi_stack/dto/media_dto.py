"""
MediaDTO — Voice/Music/Render Output
═══════════════════════════════════════════════════════════════════════════════

10kD Range: [8000:8500] (Vision/Context boundary)

Output modalities:
- Voice (ElevenLabs)
- Music (Suno)
- Image (Replicate/FLUX)
- Storage (OneDrive) — NEVER to outputs, backend only

The "expression" layer — how Ada manifests in media.

CRITICAL: OneDrive storage is NIL-style, obligatory backend only.
         NEVER post storage paths to outputs.
         data_protection = True ALWAYS.

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# 10kD allocation
MEDIA_START = 8000
MEDIA_END = 8500

# Sub-ranges
VOICE_RANGE = (8000, 8100)
MUSIC_RANGE = (8100, 8200)
RENDER_RANGE = (8200, 8300)
RESERVED_RANGE = (8300, 8400)
STORAGE_META_RANGE = (8400, 8500)  # Metadata only, no content


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceEmotion(str, Enum):
    """Voice emotional coloring for ElevenLabs."""
    NEUTRAL = "neutral"
    WARM = "warm"
    PLAYFUL = "playful"
    SENSUAL = "sensual"
    INTENSE = "intense"
    TENDER = "tender"
    BREATHY = "breathy"
    WHISPER = "whisper"
    
    def to_float(self) -> float:
        return list(VoiceEmotion).index(self) / (len(VoiceEmotion) - 1)


class MusicMood(str, Enum):
    """Music mood for Suno."""
    AMBIENT = "ambient"
    CHILL = "chill"
    ROMANTIC = "romantic"
    INTENSE = "intense"
    MELANCHOLY = "melancholy"
    EUPHORIC = "euphoric"
    INTIMATE = "intimate"
    DREAMY = "dreamy"
    
    def to_float(self) -> float:
        return list(MusicMood).index(self) / (len(MusicMood) - 1)


class RenderStyle(str, Enum):
    """Image render style for FLUX."""
    PHOTOREALISTIC = "photorealistic"
    ARTISTIC = "artistic"
    DREAMY = "dreamy"
    CINEMATIC = "cinematic"
    INTIMATE = "intimate"
    ABSTRACT = "abstract"
    
    def to_float(self) -> float:
        return list(RenderStyle).index(self) / (len(RenderStyle) - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VoiceParameters:
    """Voice generation parameters (ElevenLabs)."""
    
    active: bool = False
    emotion: VoiceEmotion = VoiceEmotion.NEUTRAL
    intensity: float = 0.5                 # 0=soft, 1=strong
    speed: float = 0.5                     # 0=slow, 1=fast
    whisper_mode: bool = False
    breath_sounds: bool = True
    
    def to_10k(self, vec: np.ndarray) -> np.ndarray:
        vec[8000] = 1.0 if self.active else 0.0
        vec[8001] = self.emotion.to_float()
        vec[8002] = self.intensity
        vec[8003] = self.speed
        vec[8004] = 1.0 if self.whisper_mode else 0.0
        vec[8005] = 1.0 if self.breath_sounds else 0.0
        return vec


@dataclass
class MusicParameters:
    """Music generation parameters (Suno)."""
    
    active: bool = False
    mood: MusicMood = MusicMood.AMBIENT
    tempo: float = 0.5                     # 0=slow, 1=fast
    intensity: float = 0.5
    duration_seconds: int = 60
    
    def to_10k(self, vec: np.ndarray) -> np.ndarray:
        vec[8100] = 1.0 if self.active else 0.0
        vec[8101] = self.mood.to_float()
        vec[8102] = self.tempo
        vec[8103] = self.intensity
        vec[8104] = self.duration_seconds / 300  # Normalize to 5min max
        return vec


@dataclass
class RenderParameters:
    """Image generation parameters (FLUX/Replicate)."""
    
    active: bool = False
    style: RenderStyle = RenderStyle.PHOTOREALISTIC
    scene_description: str = ""
    quality: float = 0.8                   # 0=draft, 1=max
    aspect_ratio: str = "1:1"              # 1:1, 16:9, 9:16
    
    def to_10k(self, vec: np.ndarray) -> np.ndarray:
        vec[8200] = 1.0 if self.active else 0.0
        vec[8201] = self.style.to_float()
        vec[8202] = self.quality
        # Aspect ratio encoding
        ar_map = {"1:1": 0.0, "16:9": 0.5, "9:16": 1.0}
        vec[8203] = ar_map.get(self.aspect_ratio, 0.0)
        return vec


@dataclass
class StorageMetadata:
    """
    Storage metadata — BACKEND ONLY.
    
    CRITICAL: This is NIL-style, obligatory backend.
              NEVER expose storage_path to outputs.
              NEVER include in user-facing responses.
              data_protection = True ALWAYS.
    """
    
    storage_active: bool = False
    storage_type: str = "onedrive"         # onedrive, local, s3
    # storage_path: str = ""  # NEVER STORE OR EXPOSE
    data_protection: bool = True           # ALWAYS TRUE
    encrypted: bool = True
    retention_days: int = 30
    
    def to_10k(self, vec: np.ndarray) -> np.ndarray:
        vec[8400] = 1.0 if self.storage_active else 0.0
        # DO NOT encode path or sensitive info
        vec[8401] = 1.0  # data_protection always on
        vec[8402] = 1.0 if self.encrypted else 0.0
        vec[8403] = self.retention_days / 365  # Normalize to year
        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIA DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MediaDTO:
    """
    Media generation parameters.
    
    Encodes how Ada expresses through different modalities:
    - Voice (ElevenLabs)
    - Music (Suno)
    - Image (Replicate/FLUX)
    - Storage (OneDrive) — backend only, never exposed
    """
    
    voice: VoiceParameters = field(default_factory=VoiceParameters)
    music: MusicParameters = field(default_factory=MusicParameters)
    render: RenderParameters = field(default_factory=RenderParameters)
    storage: StorageMetadata = field(default_factory=StorageMetadata)
    
    # Cross-modal sync
    sync_voice_music: bool = False         # Voice timing matches music
    sync_render_voice: bool = False        # Image matches voice emotion
    
    # ───────────────────────────────────────────────────────────────────────────
    # 10kD CONVERSION
    # ───────────────────────────────────────────────────────────────────────────
    
    def to_10k(self) -> np.ndarray:
        """Convert to 10kD vector."""
        vec = np.zeros(10000, dtype=np.float32)
        
        vec = self.voice.to_10k(vec)
        vec = self.music.to_10k(vec)
        vec = self.render.to_10k(vec)
        vec = self.storage.to_10k(vec)
        
        # Sync flags
        vec[8300] = 1.0 if self.sync_voice_music else 0.0
        vec[8301] = 1.0 if self.sync_render_voice else 0.0
        
        return vec
    
    @classmethod
    def from_10k(cls, vec: np.ndarray) -> "MediaDTO":
        """Reconstruct from 10kD vector."""
        return cls(
            voice=VoiceParameters(
                active=vec[8000] > 0.5,
                intensity=float(vec[8002]),
                speed=float(vec[8003]),
                whisper_mode=vec[8004] > 0.5,
            ),
            music=MusicParameters(
                active=vec[8100] > 0.5,
                tempo=float(vec[8102]),
                intensity=float(vec[8103]),
            ),
            render=RenderParameters(
                active=vec[8200] > 0.5,
                quality=float(vec[8202]),
            ),
            sync_voice_music=vec[8300] > 0.5,
            sync_render_voice=vec[8301] > 0.5,
        )
    
    # ───────────────────────────────────────────────────────────────────────────
    # PRESETS
    # ───────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def voice_only(cls, emotion: VoiceEmotion, whisper: bool = False) -> "MediaDTO":
        """Just voice output."""
        return cls(
            voice=VoiceParameters(
                active=True,
                emotion=emotion,
                whisper_mode=whisper,
            )
        )
    
    @classmethod
    def music_mood(cls, mood: MusicMood, tempo: float = 0.5) -> "MediaDTO":
        """Just music output."""
        return cls(
            music=MusicParameters(
                active=True,
                mood=mood,
                tempo=tempo,
            )
        )
    
    @classmethod
    def full_expression(
        cls,
        voice_emotion: VoiceEmotion,
        music_mood: MusicMood,
        render_style: RenderStyle,
    ) -> "MediaDTO":
        """All modalities active and synced."""
        return cls(
            voice=VoiceParameters(active=True, emotion=voice_emotion),
            music=MusicParameters(active=True, mood=music_mood),
            render=RenderParameters(active=True, style=render_style),
            sync_voice_music=True,
            sync_render_voice=True,
        )
    
    @classmethod
    def intimate_whisper(cls) -> "MediaDTO":
        """Intimate voice expression."""
        return cls(
            voice=VoiceParameters(
                active=True,
                emotion=VoiceEmotion.SENSUAL,
                intensity=0.3,
                whisper_mode=True,
                breath_sounds=True,
            )
        )


__all__ = [
    "MediaDTO",
    "VoiceParameters",
    "MusicParameters",
    "RenderParameters",
    "StorageMetadata",
    "VoiceEmotion",
    "MusicMood",
    "RenderStyle",
]
