"""
Base DTO with 10kD projection support.

UPDATED: Now imports from dimension_registry.py (single source of truth)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, TypeVar, Type
from abc import ABC, abstractmethod
import numpy as np
import hashlib

# Import canonical dimension registry
from .dimension_registry import DIMENSION_REGISTRY, get_range, DimRange

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
        """Get the dimension range for this DTO type from registry."""
        try:
            return get_range(self.dto_type)
        except KeyError:
            # Fallback for unregistered types
            return (0, TOTAL_DIMENSIONS)
    
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
    
    def to_10k(self) -> np.ndarray:
        """
        Project to full 10kD space by placing local vector in correct position.
        """
        full = np.zeros(TOTAL_DIMENSIONS, dtype=np.float32)
        local = self.to_local_vector()
        start, end = self.dimension_range
        
        # Ensure we don't overflow
        max_len = min(len(local), end - start)
        full[start:start + max_len] = local[:max_len]
        
        return full
    
    @classmethod
    @abstractmethod
    def from_local_vector(cls: Type[T], vector: np.ndarray) -> T:
        """
        Reconstruct DTO from local vector.
        """
        pass
    
    @classmethod
    def from_10k(cls: Type[T], vector: np.ndarray) -> T:
        """
        Reconstruct DTO from full 10kD space.
        """
        # Get the range for this DTO type
        try:
            start, end = get_range(cls.__name__.lower().replace('dto', ''))
        except KeyError:
            # Try without 'dto' suffix
            start, end = 0, TOTAL_DIMENSIONS
        
        local = vector[start:end]
        return cls.from_local_vector(local)
    
    def content_hash(self) -> str:
        """Generate content hash for deduplication."""
        vec = self.to_local_vector()
        return hashlib.sha256(vec.tobytes()).hexdigest()[:16]
    
    def similarity(self, other: 'BaseDTO') -> float:
        """Compute cosine similarity with another DTO."""
        v1 = self.to_local_vector()
        v2 = other.to_local_vector()
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))


# ═══════════════════════════════════════════════════════════════════════════════
# DEPRECATED — DO NOT USE
# ═══════════════════════════════════════════════════════════════════════════════

# Old DIMENSION_MAP had wrong ranges. Use dimension_registry.py instead.
# Keeping this commented for reference only:
#
# DIMENSION_MAP = {
#     "soul": (0, 2000),      # WRONG — should be (0, 500)
#     "felt": (2001, 4000),   # WRONG — should be (2000, 2400)
#     "situation": (4001, 5500),
#     "volition": (5501, 7000),
#     "vision": (7001, 8500),
#     "context": (8501, 10000),
# }


__all__ = ["BaseDTO", "DTORegistry", "TOTAL_DIMENSIONS"]
