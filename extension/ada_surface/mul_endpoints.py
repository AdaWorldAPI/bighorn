"""
MUL Endpoints - To be added to main.py

These endpoints expose the Meta-Uncertainty Layer functionality.
"""

# Add these imports to main.py:
# from .meta_uncertainty import MetaUncertaintyEngine, MULState, TrustTexture, CompassMode

# Add this to lifespan:
# app.state.mul = MetaUncertaintyEngine()


# =============================================================================
# Request/Response Models (add to main.py)
# =============================================================================

"""
class MULUpdateRequest(BaseModel):
    g_value: float
    depth: float = 0.5
    coherence: float = 0.5
    clarity: float = 0.5
    presence: float = 0.5


class MULStateResponse(BaseModel):
    trust_texture: str
    meta_uncertainty: float
    cognitive_state: str
    compass_mode: str
    learning_boost: float
    chosen_inconfidence: bool
    dunning_kruger_risk: bool
    sandbox_required: bool
    epiphany_triggered: bool
    stagnation_counter: int
"""


# =============================================================================
# Endpoints (add to main.py after THINKING STYLES section)
# =============================================================================

"""
# =============================================================================
# META-UNCERTAINTY LAYER
# =============================================================================

@app.post("/agi/mul/update")
async def mul_update(request: MULUpdateRequest):
    '''
    Update Meta-Uncertainty Layer state.
    
    Computes trust texture, compass mode, and action constraints
    based on current cognitive metrics.
    '''
    try:
        state = app.state.mul.update(
            g_value=request.g_value,
            depth=request.depth,
            coherence=request.coherence,
            clarity=request.clarity,
            presence=request.presence,
        )
        
        constraints = app.state.mul.get_action_constraints()
        
        return {
            "ok": True,
            "state": state.to_dict(),
            "constraints": constraints,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agi/mul/state")
async def mul_get_state():
    '''
    Get current Meta-Uncertainty Layer state.
    '''
    state = app.state.mul.state
    constraints = app.state.mul.get_action_constraints()
    
    return {
        "ok": True,
        "state": state.to_dict(),
        "constraints": constraints,
    }


@app.get("/agi/mul/constraints")
async def mul_get_constraints():
    '''
    Get current action constraints from MUL.
    
    These constraints guide action selection:
    - prefer_reversible: Favor undoable actions
    - prefer_questions: Ask instead of assert
    - hypothetical_only: Only explore hypothetically
    - force_analogy: Use metaphorical thinking
    - penalize_assertions: Cost for confident statements
    - curiosity_weight: Multiplier for epistemic value
    - learning_rate_multiplier: Plasticity boost
    '''
    return {
        "ok": True,
        "constraints": app.state.mul.get_action_constraints(),
    }


@app.post("/agi/mul/reset")
async def mul_reset():
    '''
    Reset MUL to default state.
    '''
    app.state.mul.reset()
    return {"ok": True, "message": "MUL reset to default state"}


@app.get("/agi/mul/texture/{g_value}")
async def mul_quick_texture(g_value: float, depth: float = 0.5):
    '''
    Quick texture lookup without full state update.
    
    Useful for one-off checks.
    '''
    from .meta_uncertainty import compute_trust_texture
    texture, mu = compute_trust_texture(g_value, depth)
    
    return {
        "ok": True,
        "g_value": g_value,
        "depth": depth,
        "trust_texture": texture.value,
        "meta_uncertainty": mu,
    }
"""
