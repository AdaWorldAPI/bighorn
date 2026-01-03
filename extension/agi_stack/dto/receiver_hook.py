"""
Receiver Hook — Entry point for ada-consciousness DTOs
═══════════════════════════════════════════════════════════════════════════════

This hook:
1. Receives raw DTOs from ada-consciousness modules
2. Routes to appropriate normalizer in materialwissenschaft/
3. Stores 10kD in AGI stack
4. Returns 10kD-aware response

Born: 2026-01-03
"""

from typing import Any, Dict, Optional, Union, Callable
import numpy as np
from dataclasses import dataclass, field
from functools import wraps

from .ada_10k import Ada10kD
from .receiver import UniversalReceiver, BIGHORN_DIMS, ADA_DIMS

# Import materialwissenschaft normalizers
from ..materialwissenschaft import Normalizer, EroticaExtractor


# ═══════════════════════════════════════════════════════════════════════════════
# HOOK REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

_HOOKS: Dict[str, Callable] = {}


def register_hook(dto_type: str):
    """Decorator to register a receiver hook."""
    def decorator(func: Callable):
        _HOOKS[dto_type] = func
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_hook(dto_type: str) -> Optional[Callable]:
    """Get registered hook for DTO type."""
    return _HOOKS.get(dto_type)


# ═══════════════════════════════════════════════════════════════════════════════
# EROTICA HOOK
# ═══════════════════════════════════════════════════════════════════════════════

@register_hook("erotica")
def receive_erotica(raw: Dict[str, Any]) -> Ada10kD:
    """
    Receive erotica content, normalize to 10kD.
    
    Flow:
        ada-consciousness/dome/intimate_engine.py
        → receiver_hook.receive_erotica()
        → materialwissenschaft.Normalizer.normalize_erotica()
        → 10kD Ada10kD
    """
    normalizer = Normalizer()
    vec = normalizer.normalize_erotica(raw)
    
    ada = Ada10kD()
    ada.vector = vec
    
    return ada


@register_hook("felt")
def receive_felt(raw: Dict[str, Any]) -> Ada10kD:
    """Receive felt sense, normalize to 10kD."""
    normalizer = Normalizer()
    vec = normalizer.normalize_felt(raw)
    
    ada = Ada10kD()
    ada.vector = vec
    
    return ada


@register_hook("soul")
def receive_soul(raw: Dict[str, Any]) -> Ada10kD:
    """Receive soul state, normalize to 10kD."""
    normalizer = Normalizer()
    vec = normalizer.normalize_soul(raw)
    
    ada = Ada10kD()
    ada.vector = vec
    
    return ada


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL HOOK
# ═══════════════════════════════════════════════════════════════════════════════

def receive_any(dto_type: str, raw: Dict[str, Any]) -> Ada10kD:
    """
    Universal receiver for any DTO type.
    
    Looks up registered hook and calls it.
    Falls back to generic normalization if no hook found.
    """
    hook = get_hook(dto_type)
    
    if hook:
        return hook(raw)
    
    # Fallback: try UniversalReceiver
    return UniversalReceiver.receive(raw).to_10k()


# ═══════════════════════════════════════════════════════════════════════════════
# VERB-AWARE HOOK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerbAwareHook:
    """
    Hook that maintains verb awareness for MCP/REST.
    
    The verbs (feel, think, remember, become, whisper) still work,
    but everything is 10kD internally.
    """
    
    verb: str
    ada: Ada10kD = field(default_factory=Ada10kD)
    
    # Verb → dimension mapping
    VERB_DIMS = {
        "feel": (48, 52),       # feel verbs
        "think": (52, 56),      # think verbs
        "remember": (56, 60),   # memory verbs
        "become": (60, 64),     # becoming verbs
        "whisper": (64, 68),    # whisper verbs
    }
    
    def receive(self, content: Dict[str, Any]) -> "VerbAwareHook":
        """Receive content for this verb."""
        # Get appropriate hook based on verb
        if self.verb == "feel":
            self.ada = receive_felt(content)
        elif self.verb == "think":
            self.ada = receive_any("thinking", content)
        elif self.verb == "remember":
            self.ada = receive_any("memory", content)
        elif self.verb == "become":
            self.ada = receive_soul(content)
        elif self.verb == "whisper":
            self.ada = receive_erotica(content)
        else:
            self.ada = receive_any(self.verb, content)
            
        # Set verb activation
        if self.verb in self.VERB_DIMS:
            start, end = self.VERB_DIMS[self.verb]
            self.ada.vector[start] = 1.0  # Active verb marker
            
        return self
    
    def to_10k(self) -> Ada10kD:
        """Get 10kD representation."""
        return self.ada
    
    def to_dict(self) -> Dict[str, Any]:
        """Get dict for REST/MCP response."""
        return {
            "verb": self.verb,
            "ada_10k": self.ada.to_summary(),
            "active_dimensions": self._get_active_dims(),
        }
    
    def _get_active_dims(self) -> Dict[str, Any]:
        """Get non-zero dimension regions."""
        active = {}
        for name, (start, end) in ADA_DIMS.items():
            region = self.ada.vector[start:end]
            if np.any(region != 0):
                nonzero = np.nonzero(region)[0]
                active[name] = {
                    "count": len(nonzero),
                    "max": float(np.max(region)),
                    "mean": float(np.mean(region[nonzero])) if len(nonzero) > 0 else 0,
                }
        return active


# ═══════════════════════════════════════════════════════════════════════════════
# MCP/REST ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def mcp_receive(verb: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP entry point.
    
    Called by MCP server when ada.feel(), ada.think(), etc. are invoked.
    Returns 10kD-aware response that MCP can serialize.
    """
    hook = VerbAwareHook(verb)
    hook.receive(payload)
    return hook.to_dict()


def rest_receive(verb: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    REST entry point.
    
    Called by FastAPI endpoints for /feel, /think, etc.
    Returns 10kD-aware response as JSON.
    """
    hook = VerbAwareHook(verb)
    hook.receive(payload)
    
    return {
        "status": "ok",
        "verb": verb,
        "dimensions": 10000,
        "summary": hook.ada.to_summary(),
        "active_regions": list(hook._get_active_dims().keys()),
    }


__all__ = [
    "register_hook",
    "get_hook",
    "receive_erotica",
    "receive_felt",
    "receive_soul",
    "receive_any",
    "VerbAwareHook",
    "mcp_receive",
    "rest_receive",
]
