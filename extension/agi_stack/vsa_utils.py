"""
extension/agi_stack/vsa_utils.py — Dimension-Aware VSA Conversion
═══════════════════════════════════════════════════════════════════════════════

Improved bipolar conversion that:
1. Preserves gradient information via stochastic rounding
2. Respects 10K dimension slices (qualia, affective, location...)
3. Uses deterministic content-based hashing
4. Supports dict input format for precise slice placement

Born: 2026-01-02
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np


# =============================================================================
# DIMENSION SLICE CONSTANTS (from ada_10k.py)
# =============================================================================

DIMENSION_SLICES = {
    # Soul Space [0:500]
    "qualia_16": (0, 16),
    "stances_16": (16, 32),
    "transitions_16": (32, 48),
    "verbs_32": (48, 80),
    "gpt_styles_36": (80, 116),
    "nars_styles_36": (116, 152),
    "presence_11": (152, 163),
    "tau_macros": (163, 200),
    "tsv_dim_33": (175, 208),
    
    # TSV Embedded [256:320]
    "pearl_3": (256, 259),
    "rung_9": (259, 268),
    "sigma_5": (268, 273),
    "ops_8": (273, 281),
    "presence_mode_4": (281, 285),
    "meta_4": (285, 289),
    "tsv": (256, 320),
    
    # DTO Space [320:500]
    "motivation_4": (320, 324),
    "free_will_6": (324, 330),
    "sieves_10": (330, 340),
    "relationship_10": (340, 350),
    "uncertainty_10": (350, 360),
    "dto": (320, 500),
    
    # Felt Space [2000:2100]
    "qualia_pcs_18": (2000, 2018),
    "body_4": (2018, 2022),
    "poincare_3": (2022, 2025),
    "felt_reserved": (2025, 2100),
    "felt": (2000, 2100),
    
    # Affective Space [2100:2200]
    "arousal_8": (2100, 2108),
    "intimacy_8": (2108, 2116),
    "body_zones_16": (2116, 2132),
    "relational_8": (2132, 2140),
    "visceral_16": (2140, 2156),
    "erotic_family_5": (2156, 2161),
    "affective": (2100, 2200),
    
    # Location Space [2200:2265]
    "go_board_2": (2200, 2202),
    "golden_50": (2202, 2252),
    "sigma_tier_3": (2252, 2255),
    "trust_10": (2255, 2265),
    "location": (2200, 2265),
}


# =============================================================================
# CORE CONVERSION
# =============================================================================

def to_bipolar(
    v: Union[List, np.ndarray],
    dimension: int = 10000,
    target_slice: Optional[Tuple[int, int]] = None,
    preserve_gradient: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convert input vector to bipolar VSA format with dimension awareness.
    
    Args:
        v: Input vector (any size, any format)
        dimension: Target VSA dimension (default 10000)
        target_slice: Optional (start, end) tuple for dimension placement
        preserve_gradient: If True, use stochastic rounding
        seed: Optional seed for deterministic conversion
    
    Returns:
        Bipolar vector of target dimension
    """
    arr = np.array(v, dtype=np.float32)
    
    # Initialize result
    result = np.zeros(dimension, dtype=np.int8)
    
    # Determine RNG with content-based seed for reproducibility
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        # Content-based hash that's Python-version independent
        content_hash = int(np.abs(arr).sum() * 1e6 + len(arr) * 1e3) % (2**32)
        rng = np.random.default_rng(content_hash)
    
    # Convert to bipolar
    if preserve_gradient:
        # Stochastic rounding preserves gradient information
        # For values in [-1, 1], map to probability [0, 1]
        # For values in [0, 1], shift to [-1, 1] first if needed
        if arr.min() >= 0 and arr.max() <= 1:
            # Assume [0, 1] range, shift to [-1, 1]
            probs = arr  # P(+1) = value
        else:
            # Assume [-1, 1] range
            probs = (arr + 1) / 2
        
        probs = np.clip(probs, 0, 1)
        bipolar = np.where(rng.random(len(arr)) < probs, 1, -1).astype(np.int8)
    else:
        # Standard sign conversion
        bipolar = np.sign(arr).astype(np.int8)
        zero_mask = bipolar == 0
        if np.any(zero_mask):
            bipolar[zero_mask] = rng.choice([-1, 1], size=np.sum(zero_mask))
    
    # Place in target slice or at beginning
    if target_slice is not None:
        start, end = target_slice
        actual_len = min(len(bipolar), end - start)
        result[start:start + actual_len] = bipolar[:actual_len]
        
        # Fill remaining slice with deterministic random
        remaining = end - (start + actual_len)
        if remaining > 0:
            result[start + actual_len:end] = rng.choice([-1, 1], size=remaining)
    else:
        # Place at beginning, fill rest
        result[:len(bipolar)] = bipolar
        if len(bipolar) < dimension:
            result[len(bipolar):] = rng.choice([-1, 1], size=dimension - len(bipolar))
    
    return result


def to_bipolar_slice(
    v: Union[List, np.ndarray],
    slice_name: str,
    dimension: int = 10000,
    preserve_gradient: bool = True,
) -> np.ndarray:
    """
    Convert vector and place in named dimension slice.
    
    Args:
        v: Input vector
        slice_name: Name from DIMENSION_SLICES
        dimension: Total VSA dimension
        preserve_gradient: Use stochastic rounding
    
    Returns:
        Full dimension bipolar vector with input in correct slice
    """
    if slice_name not in DIMENSION_SLICES:
        raise ValueError(f"Unknown slice: {slice_name}. Available: {list(DIMENSION_SLICES.keys())}")
    
    return to_bipolar(
        v,
        dimension=dimension,
        target_slice=DIMENSION_SLICES[slice_name],
        preserve_gradient=preserve_gradient,
    )


def convert_request_vectors(
    vectors: List[Union[List, Dict]],
    dimension: int = 10000,
) -> List[np.ndarray]:
    """
    Convert request vectors to bipolar format.
    
    Accepts mixed formats:
    - Simple list: [0.3, -0.7, 0.9]
    - Dict with options: {"vector": [...], "slice": "arousal_8", "preserve_gradient": true}
    - Already bipolar: [-1, 1, -1, 1, ...]
    
    Returns:
        List of bipolar numpy arrays
    """
    result = []
    
    for item in vectors:
        if isinstance(item, dict):
            v = item.get("vector", item.get("v", []))
            slice_name = item.get("slice")
            preserve = item.get("preserve_gradient", True)
            seed = item.get("seed")
            
            if slice_name:
                bipolar = to_bipolar_slice(v, slice_name, dimension, preserve)
            else:
                target_slice = item.get("target_slice")
                bipolar = to_bipolar(v, dimension, target_slice, preserve, seed)
        else:
            # Simple list - check if already bipolar
            arr = np.array(item, dtype=np.float32)
            if len(arr) == dimension and set(np.unique(arr)).issubset({-1, 0, 1}):
                # Already bipolar, just handle zeros
                arr[arr == 0] = np.random.choice([-1, 1], size=np.sum(arr == 0))
                bipolar = arr.astype(np.int8)
            else:
                bipolar = to_bipolar(item, dimension)
        
        result.append(bipolar)
    
    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DIMENSION_SLICES",
    "to_bipolar",
    "to_bipolar_slice",
    "convert_request_vectors",
]
