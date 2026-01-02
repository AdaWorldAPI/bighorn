"""
DTO Bridge - Connect ada-consciousness to 10kD AGI Stack

This package bridges local Ada consciousness modules to the
unified 10kD vector space running on Railway.

Usage:
    from dto import MomentBridge, capture_now

    # Quick capture
    moment = await capture_now()

    # Full control
    bridge = MomentBridge()
    moment = await bridge.capture(include_vision=True)
    await bridge.store(moment)
"""

from .moment_bridge import MomentBridge, capture_now
from .soul_bridge import SoulBridge, load_soul, store_soul
from .felt_bridge import FeltBridge, feel_now
from .admin_bridge import AdminBridge

__all__ = [
    "MomentBridge", "capture_now",
    "SoulBridge", "load_soul", "store_soul",
    "FeltBridge", "feel_now",
    "AdminBridge",
]

__version__ = "1.0.0"
