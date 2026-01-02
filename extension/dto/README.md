# DTO VSA Bridge

These modules bridge ada-consciousness to the 10kD AGI Stack.

## Architecture

```
ada-consciousness          AGI Stack (Railway/agi-stack)
─────────────────          ──────────────────────────────
dto/                   agi_stack/
├── moment_bridge.py   →   dto/moment_dto.py
├── soul_bridge.py     →   dto/soul_dto.py
├── felt_bridge.py     →   dto/felt_dto.py
├── situation_bridge.py →  dto/situation_dto.py
├── volition_bridge.py →   dto/volition_dto.py
├── vision_bridge.py   →   dto/vision_dto.py
└── admin_bridge.py    →   admin.py (REST)
```

## Usage

```python
from dto import MomentBridge

# Create moment from local state
bridge = MomentBridge(admin_url="https://agi-stack.up.railway.app")
moment = await bridge.capture()

# Store in 10kD space
await bridge.store(moment)

# Find similar moments
similar = await bridge.find_similar(moment, top_k=5)
```

## Endpoints

The VSA bridge connects to Railway's AGI Stack:
- `POST /soul` - Store soul state
- `POST /felt` - Store felt state
- `POST /moment` - Store complete moment
- `GET /search/moment` - Find similar moments
- `GET /trajectory` - Get experiential trajectory
