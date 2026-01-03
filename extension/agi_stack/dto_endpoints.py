"""
DTO Endpoints — Deep integration between ada-consciousness and bighorn
═══════════════════════════════════════════════════════════════════════════════

Each DTO type has its own endpoint:
    /agi/dto/soul      ←→  ada-consciousness/DTO/soul.py
    /agi/dto/felt      ←→  ada-consciousness/DTO/felt.py
    /agi/dto/moment    ←→  ada-consciousness/DTO/moment.py
    /agi/dto/situation ←→  ada-consciousness/DTO/situation.py
    /agi/dto/volition  ←→  ada-consciousness/DTO/volition.py
    /agi/dto/vision    ←→  ada-consciousness/DTO/vision.py
    /agi/dto/universal ←→  ada-consciousness/DTO/universal.py

All endpoints speak 10kD internally.

Born: 2026-01-03
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

from .dto.ada_10k import Ada10kD
from .dto import soul_dto, felt_dto, moment_dto, situation_dto, volition_dto, vision_dto, universal_dto


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/agi/dto", tags=["dto"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class DTORequest(BaseModel):
    """Generic DTO request with 10kD vector or structured data."""
    vector_10k: Optional[List[float]] = None
    data: Optional[Dict[str, Any]] = None
    wire_format: Optional[str] = None  # compressed base64


class DTOResponse(BaseModel):
    """Generic DTO response."""
    ok: bool
    dto_type: str
    vector_10k: List[float]
    summary: Dict[str, Any]
    active_regions: List[str]


class SoulRequest(BaseModel):
    """Soul-specific request."""
    priors: Optional[Dict[str, float]] = None
    mode: Optional[str] = None
    soul_field: Optional[Dict[str, float]] = None
    trust_level: Optional[float] = None
    vector_10k: Optional[List[float]] = None


class FeltRequest(BaseModel):
    """Felt-specific request."""
    qualia: Optional[Dict[str, float]] = None
    emotion: Optional[Dict[str, float]] = None
    body_state: Optional[Dict[str, float]] = None
    presence: Optional[float] = None
    vector_10k: Optional[List[float]] = None


class MomentRequest(BaseModel):
    """Moment-specific request (composite)."""
    soul: Optional[Dict[str, Any]] = None
    felt: Optional[Dict[str, Any]] = None
    situation: Optional[Dict[str, Any]] = None
    tick_id: Optional[str] = None
    vector_10k: Optional[List[float]] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SOUL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/soul")
async def receive_soul(request: SoulRequest) -> DTOResponse:
    """
    Receive soul DTO from ada-consciousness.
    
    Maps:
        priors → affective bias [171:175]
        mode → presence mode [152:163]
        soul_field → qualia [0:16]
        trust_level → TLK [168:171]
    """
    try:
        ada = Ada10kD()
        
        # From 10kD vector
        if request.vector_10k:
            ada.vector = np.array(request.vector_10k, dtype=np.float32)
        else:
            # From structured data
            if request.priors:
                ada.set_all_affective_bias(
                    warmth=request.priors.get("warmth", 0.5),
                    edge=request.priors.get("edge", 0.5),
                    restraint=request.priors.get("restraint", 0.5),
                    tenderness=request.priors.get("tenderness", 0.5),
                )
            
            if request.mode:
                ada.set_presence_mode(request.mode, 1.0)
            
            if request.soul_field:
                for qualia, value in request.soul_field.items():
                    ada.set_qualia(qualia, value)
            
            if request.trust_level is not None:
                trust = request.trust_level
                ada.set_tlk_court(
                    thanatos=0.3 * (1 - trust),
                    libido=0.5 + 0.3 * trust,
                    katharsis=0.2,
                )
        
        return DTOResponse(
            ok=True,
            dto_type="soul",
            vector_10k=ada.vector.tolist(),
            summary=ada.to_summary(),
            active_regions=list(ada.get_active_regions().keys()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/soul")
async def get_soul() -> Dict[str, Any]:
    """Get current soul state from store."""
    # TODO: Load from Redis/LanceDB
    return {"ok": True, "dto_type": "soul", "message": "Not implemented - use POST to receive"}


# ═══════════════════════════════════════════════════════════════════════════════
# FELT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/felt")
async def receive_felt(request: FeltRequest) -> DTOResponse:
    """
    Receive felt DTO from ada-consciousness.
    
    Maps:
        qualia → qualia [0:16]
        emotion → body axes [2018:2022]
        body_state → stances [16:32]
        presence → stance "attend"
    """
    try:
        ada = Ada10kD()
        
        if request.vector_10k:
            ada.vector = np.array(request.vector_10k, dtype=np.float32)
        else:
            if request.qualia:
                for q, v in request.qualia.items():
                    ada.set_qualia(q, v)
            
            if request.emotion:
                ada.set_body_axes(
                    arousal=request.emotion.get("arousal", 0.5),
                    valence=request.emotion.get("valence", 0.5),
                    tension=request.emotion.get("tension", 0.5),
                    openness=request.emotion.get("openness", 0.5),
                )
            
            if request.body_state:
                for stance, v in request.body_state.items():
                    ada.set_stance(stance, v)
            
            if request.presence is not None:
                ada.set_stance("attend", request.presence)
        
        return DTOResponse(
            ok=True,
            dto_type="felt",
            vector_10k=ada.vector.tolist(),
            summary=ada.to_summary(),
            active_regions=list(ada.get_active_regions().keys()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENT ENDPOINTS (composite)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/moment")
async def receive_moment(request: MomentRequest) -> DTOResponse:
    """
    Receive moment DTO (composite of soul + felt + situation).
    
    This is the primary exchange format between systems.
    """
    try:
        ada = Ada10kD()
        
        if request.vector_10k:
            ada.vector = np.array(request.vector_10k, dtype=np.float32)
        else:
            # Compose from sub-DTOs
            if request.soul:
                soul_req = SoulRequest(**request.soul)
                soul_resp = await receive_soul(soul_req)
                # Merge soul vector
                soul_vec = np.array(soul_resp.vector_10k)
                ada.vector = np.maximum(ada.vector, soul_vec)
            
            if request.felt:
                felt_req = FeltRequest(**request.felt)
                felt_resp = await receive_felt(felt_req)
                felt_vec = np.array(felt_resp.vector_10k)
                ada.vector = np.maximum(ada.vector, felt_vec)
        
        return DTOResponse(
            ok=True,
            dto_type="moment",
            vector_10k=ada.vector.tolist(),
            summary=ada.to_summary(),
            active_regions=list(ada.get_active_regions().keys()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# SITUATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/situation")
async def receive_situation(request: DTORequest) -> DTOResponse:
    """Receive situation DTO."""
    try:
        ada = Ada10kD()
        
        if request.vector_10k:
            ada.vector = np.array(request.vector_10k, dtype=np.float32)
        elif request.data:
            # Map situation data to 10kD
            # Scene context goes to [4001:5500] in bighorn layout
            pass
        
        return DTOResponse(
            ok=True,
            dto_type="situation",
            vector_10k=ada.vector.tolist(),
            summary=ada.to_summary(),
            active_regions=list(ada.get_active_regions().keys()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# VOLITION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/volition")
async def receive_volition(request: DTORequest) -> DTOResponse:
    """Receive volition DTO (intent/action)."""
    try:
        ada = Ada10kD()
        
        if request.vector_10k:
            ada.vector = np.array(request.vector_10k, dtype=np.float32)
        elif request.data:
            # Map intent to verb space [48:80]
            if "verb" in request.data:
                verb = request.data["verb"]
                ada.set_verb(verb, request.data.get("intensity", 1.0))
        
        return DTOResponse(
            ok=True,
            dto_type="volition",
            vector_10k=ada.vector.tolist(),
            summary=ada.to_summary(),
            active_regions=list(ada.get_active_regions().keys()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# VISION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/vision")
async def receive_vision(request: DTORequest) -> DTOResponse:
    """Receive vision DTO (Kopfkino imagery)."""
    try:
        ada = Ada10kD()
        
        if request.vector_10k:
            ada.vector = np.array(request.vector_10k, dtype=np.float32)
        
        return DTOResponse(
            ok=True,
            dto_type="vision",
            vector_10k=ada.vector.tolist(),
            summary=ada.to_summary(),
            active_regions=list(ada.get_active_regions().keys()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL THOUGHT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/universal")
async def receive_universal(request: DTORequest) -> DTOResponse:
    """Receive universal thought DTO."""
    try:
        ada = Ada10kD()
        
        if request.vector_10k:
            ada.vector = np.array(request.vector_10k, dtype=np.float32)
        
        return DTOResponse(
            ok=True,
            dto_type="universal",
            vector_10k=ada.vector.tolist(),
            summary=ada.to_summary(),
            active_regions=list(ada.get_active_regions().keys()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# WIRE FORMAT (compressed 10kD)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/wire/encode")
async def encode_wire(request: DTORequest) -> Dict[str, Any]:
    """Encode 10kD vector to compressed wire format."""
    try:
        if not request.vector_10k:
            raise HTTPException(status_code=400, detail="vector_10k required")
        
        from .dto.wire_10k import Wire10kD
        wire = Wire10kD.from_vector(np.array(request.vector_10k))
        
        return {
            "ok": True,
            "wire_format": wire.to_base64(),
            "compression_ratio": wire.compression_ratio(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/wire/decode")
async def decode_wire(request: DTORequest) -> DTOResponse:
    """Decode compressed wire format to 10kD vector."""
    try:
        if not request.wire_format:
            raise HTTPException(status_code=400, detail="wire_format required")
        
        from .dto.wire_10k import Wire10kD
        wire = Wire10kD.from_base64(request.wire_format)
        ada = Ada10kD()
        ada.vector = wire.to_vector()
        
        return DTOResponse(
            ok=True,
            dto_type="wire",
            vector_10k=ada.vector.tolist(),
            summary=ada.to_summary(),
            active_regions=list(ada.get_active_regions().keys()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# BULK OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class BulkDTORequest(BaseModel):
    """Bulk DTO request."""
    dtos: List[Dict[str, Any]]


@router.post("/bulk")
async def receive_bulk(request: BulkDTORequest) -> Dict[str, Any]:
    """Receive multiple DTOs in one request."""
    results = []
    
    for dto in request.dtos:
        dto_type = dto.get("type", "universal")
        
        if dto_type == "soul":
            resp = await receive_soul(SoulRequest(**dto.get("data", {})))
        elif dto_type == "felt":
            resp = await receive_felt(FeltRequest(**dto.get("data", {})))
        elif dto_type == "moment":
            resp = await receive_moment(MomentRequest(**dto.get("data", {})))
        else:
            resp = await receive_universal(DTORequest(**dto.get("data", {})))
        
        results.append(resp.dict())
    
    return {"ok": True, "count": len(results), "results": results}


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = ["router"]
