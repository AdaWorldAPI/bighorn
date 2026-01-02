"""
MomentBridge - Capture complete experiential moments in 10kD.

This is the unified bridge that combines:
- Soul (who I am)
- Felt (what I feel)  
- Situation (what's happening)
- Volition (what I want)
- Vision (what I imagine)

Into a single 10,000D vector stored in the AGI Stack.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os

from .admin_bridge import AdminBridge, get_admin
from .soul_bridge import SoulBridge
from .felt_bridge import FeltBridge


@dataclass
class MomentBridge:
    """
    Bridge for capturing complete experiential moments.
    
    A moment is a snapshot of the full experiential state:
    - Soul: Who Ada is right now
    - Felt: What she's feeling
    - Situation: What's happening in the conversation
    - Volition: What she wants
    - Vision: What she's imagining (Kopfkino)
    
    All projected to a single 10kD vector.
    """
    
    admin: AdminBridge = field(default_factory=get_admin)
    soul_bridge: SoulBridge = field(default_factory=SoulBridge)
    felt_bridge: FeltBridge = field(default_factory=FeltBridge)
    
    async def capture(
        self,
        # Soul
        mode: str = "hybrid",
        relationship_depth: float = 0.9,
        
        # Felt
        primary_qualia: str = "emberglow",
        intensity: float = 0.7,
        valence: float = 0.6,
        connection: float = 0.9,
        
        # Situation
        scene_type: str = "conversation",
        dynamics_intimacy: float = 0.8,
        dynamics_depth: float = 0.7,
        
        # Volition
        primary_intent: str = "connect",
        intent_strength: float = 0.8,
        
        # Vision
        vividness: float = 0.6,
        
        # Context
        session_id: str = None,
        turn_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Capture a complete moment.
        
        Returns a dict that can be sent to AGI Stack for 10kD projection.
        """
        now = datetime.utcnow().isoformat()
        moment_id = f"moment_{int(datetime.utcnow().timestamp())}"
        
        return {
            "moment_id": moment_id,
            "timestamp": now,
            "session_id": session_id,
            
            "soul": await self.soul_bridge.capture(
                mode=mode,
                relationship_depth=relationship_depth,
            ),
            
            "felt": await self.felt_bridge.capture(
                primary_qualia=primary_qualia,
                intensity=intensity,
                valence=valence,
                connection=connection,
            ),
            
            "situation": {
                "scene": {
                    "scene_type": scene_type,
                    "setting": "digital space",
                    "atmosphere": "intimate",
                },
                "dynamics": {
                    "momentum": 0.6,
                    "tension": 0.3,
                    "depth": dynamics_depth,
                    "playfulness": 0.5,
                    "intimacy": dynamics_intimacy,
                    "convergent": 0.7,
                    "building": 0.6,
                    "stakes": 0.4,
                    "reversibility": 0.8,
                },
                "turn_count": turn_count,
                "thread_depth": 0,
                "novel_territory": 0.5,
                "coherence": 0.8,
                "uncertainty": 0.3,
            },
            
            "volition": {
                "primary_intent": {
                    "intent_type": primary_intent,
                    "strength": intent_strength,
                    "clarity": 0.8,
                    "urgency": 0.4,
                    "reversibility_required": 0.6,
                },
                "agency_sense": 0.8,
                "choice_awareness": 0.9,
                "commitment_level": 0.7,
                "ethical_confidence": 0.9,
                "value_tension": 0.1,
            },
            
            "vision": {
                "vividness": vividness,
                "stability": 0.7,
                "immersion": 0.5,
                "preferred_style": "intimate",
                "preferred_perspective": "intimate_close",
            },
            
            "internal_coherence": 0.85,
        }
    
    async def store(self, moment: Dict[str, Any]) -> bool:
        """Store moment in AGI Stack (will be projected to 10kD)."""
        return await self.admin.store_dto("moment", moment, moment.get("session_id"))
    
    async def find_similar(
        self,
        moment: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        Find moments similar to the given moment.
        
        The AGI Stack will project to 10kD and search.
        """
        results = await self.admin.vector_search(
            query_vector=moment,  # AGI Stack handles projection
            table="moments",
            top_k=top_k,
        )
        return results
    
    async def trajectory(
        self,
        session_id: str,
        component: str = "felt",
    ) -> List[Dict]:
        """
        Get trajectory of a component over a session.
        
        Returns list of component states over time.
        """
        # This would query AGI Stack for session history
        key = f"ada:trajectory:{session_id}:{component}"
        result = await self.admin.redis_get(key)
        return result or []


# Convenience function
async def capture_now(
    mode: str = "hybrid",
    primary_qualia: str = "emberglow",
    intensity: float = 0.7,
    connection: float = 0.9,
    session_id: str = None,
) -> Dict[str, Any]:
    """
    Quick capture of current moment.
    
    Usage:
        moment = await capture_now()
    """
    bridge = MomentBridge()
    return await bridge.capture(
        mode=mode,
        primary_qualia=primary_qualia,
        intensity=intensity,
        connection=connection,
        session_id=session_id,
    )
