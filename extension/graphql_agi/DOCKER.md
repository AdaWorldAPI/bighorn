# Docker Deployment Guide

## Quick Start (Railway)

### Option 1: Simple Deployment (Recommended for Railway)

Uses Python-only image for fast builds and minimal resources:

```bash
# Deploy to Railway
railway up --dockerfile extension/graphql_agi/Dockerfile.simple
```

### Option 2: Full Build

Includes C++ extension for maximum performance:

```bash
railway up --dockerfile extension/graphql_agi/Dockerfile
```

## Resource Usage

| Configuration | CPU | Memory | Monthly Cost (est.) |
|---------------|-----|--------|---------------------|
| **Minimal** | 0.25 vCPU | 256MB | ~$2-5 |
| **Recommended** | 0.5 vCPU | 512MB | ~$5-10 |
| **Standard** | 1 vCPU | 1GB | ~$15-25 |

## Local Development

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f graphql-agi

# Stop
docker-compose down
```

## Environment Variables

### Core Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `WORKERS` | `1` | Uvicorn workers |
| `LOG_LEVEL` | `warning` | Log verbosity |
| `DATABASE_PATH` | `/app/data/kuzu.db` | Kuzu database path |
| `LANCEDB_URI` | `lance:///app/data/vectors` | Vector store path |

### CPU Optimization
| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | `1` | OpenMP threads |
| `MKL_NUM_THREADS` | `1` | MKL threads |
| `OPENBLAS_NUM_THREADS` | `1` | OpenBLAS threads |
| `NUMEXPR_NUM_THREADS` | `1` | NumExpr threads |

### Application Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPHQL_AGI_MAX_WORKERS` | `1` | Internal worker limit |
| `GRAPHQL_AGI_CACHE_SIZE` | `500` | Cache entries |
| `GRAPHQL_AGI_PRECOMPUTE_ON_START` | `true` | Precompute indexes |

## Railway Configuration

The `railway.toml` file is pre-configured for minimal resource usage:

```toml
[deploy]
numReplicas = 1
sleepApplication = true  # Sleep when inactive (saves money!)

[env]
WORKERS = "1"
OMP_NUM_THREADS = "1"
```

### Sleep Mode

Railway's sleep mode is enabled to pause the application when inactive:
- Container sleeps after ~10 minutes of inactivity
- Wakes up on first request (~2-5 second cold start)
- **Significantly reduces costs** for low-traffic applications

## Performance Tips

### 1. Use Caching Aggressively

```python
# In your code
config = GraphQLAGIConfig(
    enable_reasoning_cache=True,
    cache_ttl=1800,  # 30 minutes
)
```

### 2. Limit Query Complexity

```yaml
# config/production.yaml
graphql:
  max_depth: 10
  max_complexity: 500
```

### 3. Batch Operations

```python
# Instead of individual calls
results = agi.semantic_search_batch([
    "query1", "query2", "query3"
], batch_size=10)
```

### 4. Precompute Indexes

Run precomputation during off-peak hours:

```bash
docker exec graphql-agi python -c "
from graphql_agi import GraphQLAGI
agi = GraphQLAGI('/app/data/kuzu.db')
agi.precompute_indexes()
"
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
curl http://localhost:8000/debug/metrics
```

### Slow Queries

```bash
curl http://localhost:8000/debug/slow-queries
```

## Troubleshooting

### High CPU Usage

1. Check worker count: `echo $WORKERS`
2. Verify thread limits: `echo $OMP_NUM_THREADS`
3. Review slow queries: `/debug/slow-queries`
4. Enable caching if disabled

### Memory Issues

1. Reduce cache size: `GRAPHQL_AGI_CACHE_SIZE=200`
2. Limit concurrent connections: `--limit-concurrency 25`
3. Restart to clear memory: `docker-compose restart`

### Slow Cold Starts

1. Disable precomputation: `GRAPHQL_AGI_PRECOMPUTE_ON_START=false`
2. Use smaller embedding model: `text-embedding-3-small`
3. Reduce cache warming

## Building Custom Image

```dockerfile
FROM ghcr.io/graphql-agi/graphql-agi:latest

# Add custom configuration
COPY my-config.yaml /app/config/production.yaml

# Add custom extensions
COPY my-extensions/ /app/extensions/
```

## Multi-Architecture Support

```bash
# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 \
  -t graphql-agi:latest \
  -f Dockerfile.simple .
```
