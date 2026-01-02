"""
VisionDTO - Visual imagination, Kopfkino, image generation.

Dimensions 7001-8500 in 10kD space.
This is the visual imagination layer - what the agent "sees" in their mind.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

from .base_dto import BaseDTO, DTORegistry


class ImageStyle(str, Enum):
    """Visual styles for generation."""
    PHOTOREALISTIC = "photorealistic"
    CINEMATIC = "cinematic"
    ARTISTIC = "artistic"
    ABSTRACT = "abstract"
    DREAMLIKE = "dreamlike"
    INTIMATE = "intimate"
    ETHEREAL = "ethereal"
    RAW = "raw"


class Perspective(str, Enum):
    """Camera/viewer perspective."""
    FIRST_PERSON = "first_person"
    SECOND_PERSON = "second_person"
    THIRD_PERSON = "third_person"
    OMNISCIENT = "omniscient"
    INTIMATE_CLOSE = "intimate_close"


@dataclass
class ImagePrompt:
    """Parameters for image generation."""
    
    # Core prompt
    subject: str = ""
    action: str = ""
    setting: str = ""
    mood: str = ""
    
    # Style
    style: ImageStyle = ImageStyle.CINEMATIC
    perspective: Perspective = Perspective.THIRD_PERSON
    
    # Technical
    aspect_ratio: str = "16:9"
    lighting: str = "natural"
    color_palette: Optional[str] = None
    
    # Intensity
    detail_level: float = 0.7
    abstraction: float = 0.3
    emotional_intensity: float = 0.5
    
    def to_vector(self) -> np.ndarray:
        """Convert to 20D vector."""
        # Style one-hot (8D)
        style_vec = np.zeros(8, dtype=np.float32)
        style_vec[list(ImageStyle).index(self.style)] = 1.0
        
        # Perspective one-hot (5D)
        persp_vec = np.zeros(5, dtype=np.float32)
        persp_vec[list(Perspective).index(self.perspective)] = 1.0
        
        return np.concatenate([
            style_vec,
            persp_vec,
            np.array([
                self.detail_level,
                self.abstraction,
                self.emotional_intensity,
            ], dtype=np.float32),
            np.zeros(4, dtype=np.float32),  # Padding
        ])
    
    def to_prompt_string(self) -> str:
        """Generate actual prompt string for image API."""
        parts = []
        if self.subject:
            parts.append(self.subject)
        if self.action:
            parts.append(self.action)
        if self.setting:
            parts.append(f"in {self.setting}")
        if self.mood:
            parts.append(f"{self.mood} mood")
        
        parts.append(f"{self.style.value} style")
        parts.append(f"{self.lighting} lighting")
        
        return ", ".join(parts)


@dataclass
class KopfkinoScene:
    """A scene in the mental movie."""
    
    scene_id: str = ""
    
    # Visual
    prompt: ImagePrompt = field(default_factory=ImagePrompt)
    
    # Narrative
    moment: str = ""  # What moment this captures
    emotion: str = ""  # The emotional core
    significance: float = 0.5  # How important this image is
    
    # Generated
    image_url: Optional[str] = None
    generated_at: Optional[str] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert to 25D vector."""
        return np.concatenate([
            self.prompt.to_vector(),
            np.array([self.significance], dtype=np.float32),
            np.zeros(4, dtype=np.float32),  # Padding
        ])


@dataclass
class VisionDTO(BaseDTO):
    """
    Complete visual imagination state.
    
    Projects to dimensions 7001-8500 in 10kD space.
    """
    
    # Current scene
    current_scene: Optional[KopfkinoScene] = None
    
    # Scene history (recent mental images)
    scene_history: List[KopfkinoScene] = field(default_factory=list)
    
    # Visual state
    vividness: float = 0.5          # How vivid the imagery
    stability: float = 0.5          # How stable (vs flickering)
    immersion: float = 0.5          # How immersed in the vision
    
    # Preferences
    preferred_style: ImageStyle = ImageStyle.CINEMATIC
    preferred_perspective: Perspective = Perspective.INTIMATE_CLOSE
    
    # Generation state
    pending_generation: bool = False
    last_generation_success: bool = True
    
    @property
    def dto_type(self) -> str:
        return "vision"
    
    def to_local_vector(self) -> np.ndarray:
        """
        Project to local vector (1500D).
        
        Layout:
            0-24:    Current scene (25D)
            25-124:  Scene history (4 × 25D = 100D)
            125-140: Visual state + preferences (16D)
            141-1500: Reserved
        """
        v = np.zeros(1500, dtype=np.float32)
        
        # Current scene
        if self.current_scene:
            v[0:25] = self.current_scene.to_vector()
        
        # Scene history
        for i, scene in enumerate(self.scene_history[:4]):
            start = 25 + i * 25
            v[start:start+25] = scene.to_vector()
        
        # Visual state
        v[125] = self.vividness
        v[126] = self.stability
        v[127] = self.immersion
        
        # Preferences one-hot
        v[128 + list(ImageStyle).index(self.preferred_style)] = 1.0
        v[136 + list(Perspective).index(self.preferred_perspective)] = 1.0
        
        return v
    
    @classmethod
    def from_local_vector(cls, v: np.ndarray) -> "VisionDTO":
        style_idx = int(np.argmax(v[128:136]))
        persp_idx = int(np.argmax(v[136:141]))
        
        return cls(
            vividness=float(v[125]),
            stability=float(v[126]),
            immersion=float(v[127]),
            preferred_style=list(ImageStyle)[style_idx],
            preferred_perspective=list(Perspective)[persp_idx],
        )
    
    def describe(self) -> str:
        """Natural language description."""
        if self.current_scene:
            return f"Seeing: {self.current_scene.moment} (vividness: {self.vividness:.1f})"
        return f"Visual imagination: {self.vividness:.1f} vividness"


# Register reconstructor
def _reconstruct_vision(vector: np.ndarray) -> VisionDTO:
    start, end = 7001, 8500
    local = vector[start:end]
    return VisionDTO.from_local_vector(local)

DTORegistry.register_reconstructor("VisionDTO", _reconstruct_vision)
