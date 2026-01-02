"""
Base DTO with 10kD projection support.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, TypeVar, Type
from abc import ABC, abstractmethod
import numpy as np
import hashlib

# 10kD dimension allocation
DIMENSION_MAP = {
    "soul": (0, 2000),
    "felt": (2001, 4000),
    "situation": (4001, 5500),
    "volition": (5501, 7000),
    "vision": (7001, 8500),
    "context": (8501, 10000),
}

TOTAL_DIMENSIONS = 10000

T = TypeVar('T', bound='BaseDTO')


class DTORegistry:
    """Registry for DTO types and their projectors."""
    
    _projectors: Dict[str, callable] = {}
    _reconstructors: Dict[str, callable] = {}
    
    @classmethod
    def register_projector(cls, dto_type: str, projector: callable):
        cls._projectors[dto_type] = projector
    
    @classmethod
    def register_reconstructor(cls, dto_type: str, reconstructor: callable):
        cls._reconstructors[dto_type] = reconstructor
    
    @classmethod
    def get_projector(cls, dto_type: str) -> Optional[callable]:
        return cls._projectors.get(dto_type)
    
    @classmethod
    def get_reconstructor(cls, dto_type: str) -> Optional[callable]:
        return cls._reconstructors.get(dto_type)


@dataclass
class BaseDTO(ABC):
    """
    Base class for all DTOs with 10kD projection support.
    """
    
    @property
    @abstractmethod
    def dto_type(self) -> str:
        """Return the DTO type (soul, felt, situation, etc)."""
        pass
    
    @property
    def dimension_range(self) -> tuple:
        """Get the dimension range for this DTO type."""
        return DIMENSION_MAP.get(self.dto_type, (0, TOTAL_DIMENSIONS))
    
    @property
    def dimension_count(self) -> int:
        """Number of dimensions allocated to this DTO type."""
        start, end = self.dimension_range
        return end - start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)
    
    @abstractmethod
    def to_local_vector(self) -> np.ndarray:
        """
        Project to local vector space (within allocated dimensions).
        Subclasses implement this to define their projection logic.
        """
        pass
    
    def to_10kd(self) -> np.ndarray:
        """
        Project to full 10kD space.
        Places local vector in correct dimension range.
        """
        full = np.zeros(TOTAL_DIMENSIONS, dtype=np.float32)
        start, end = self.dimension_range
        
        local = self.to_local_vector()
        
        # Pad or truncate to fit
        target_size = end - start
        if len(local) < target_size:
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:len(local)] = local
            local = padded
        elif len(local) > target_size:
            local = local[:target_size]
        
        full[start:end] = local
        return full
    
    @classmethod
    def from_10kd(cls: Type[T], vector: np.ndarray) -> T:
        """
        Reconstruct from 10kD vector (lossy).
        Extracts relevant dimension range and reconstructs.
        """
        reconstructor = DTORegistry.get_reconstructor(cls.__name__)
        if reconstructor:
            return reconstructor(vector)
        raise NotImplementedError(f"No reconstructor for {cls.__name__}")
    
    def blend(self: T, other: T, alpha: float = 0.5) -> T:
        """
        Blend with another DTO of same type.
        alpha=0 → self, alpha=1 → other
        """
        # Default: blend in vector space, then reconstruct
        v1 = self.to_local_vector()
        v2 = other.to_local_vector()
        blended = v1 * (1 - alpha) + v2 * alpha
        
        # This is lossy but provides default behavior
        return self  # Subclasses should override for proper blending
    
    def fingerprint(self) -> str:
        """Generate a short fingerprint for this DTO state."""
        data = str(self.to_dict()).encode()
        return hashlib.md5(data).hexdigest()[:8]
    
    def similarity(self, other: 'BaseDTO') -> float:
        """Compute cosine similarity with another DTO."""
        v1 = self.to_10kd()
        v2 = other.to_10kd()
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
