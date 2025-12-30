#!/bin/bash
# =============================================================================
# GraphQL AGI - Docker Entrypoint
# =============================================================================
# Optimized for Railway and low-resource environments
# =============================================================================

set -e

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# -----------------------------------------------------------------------------
# Resource optimization
# -----------------------------------------------------------------------------

# Limit CPU affinity if running on multi-core (optional)
if command -v taskset &> /dev/null && [ -n "${CPU_AFFINITY:-}" ]; then
    log_info "Setting CPU affinity to: $CPU_AFFINITY"
    exec taskset -c "$CPU_AFFINITY" "$@"
fi

# Set process priority (nice level)
NICE_LEVEL="${NICE_LEVEL:-10}"
if [ "$NICE_LEVEL" != "0" ]; then
    log_info "Setting nice level to: $NICE_LEVEL"
fi

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------

# Default values
export DATABASE_PATH="${DATABASE_PATH:-/app/data/kuzu.db}"
export LANCEDB_URI="${LANCEDB_URI:-lance:///app/data/vectors}"
export PORT="${PORT:-8000}"
export WORKERS="${WORKERS:-1}"
export LOG_LEVEL="${LOG_LEVEL:-warning}"

# Ensure data directories exist
mkdir -p /app/data /app/cache /app/logs 2>/dev/null || true

# -----------------------------------------------------------------------------
# Precomputation (if enabled)
# -----------------------------------------------------------------------------

if [ "${GRAPHQL_AGI_PRECOMPUTE_ON_START:-true}" = "true" ]; then
    log_info "Running precomputation pipeline..."

    python -c "
from graphql_agi.core.client import GraphQLAGI, GraphQLAGIConfig
import os

config = GraphQLAGIConfig(
    database_path=os.environ.get('DATABASE_PATH', '/app/data/kuzu.db'),
    lancedb_uri=os.environ.get('LANCEDB_URI', 'lance:///app/data/vectors'),
)

try:
    client = GraphQLAGI(config=config)
    print('Precomputation complete')
    client.close()
except Exception as e:
    print(f'Precomputation skipped: {e}')
" 2>/dev/null || log_warn "Precomputation skipped (no existing data)"

fi

# -----------------------------------------------------------------------------
# Memory optimization
# -----------------------------------------------------------------------------

# Set Python garbage collection thresholds
export PYTHONGC="${PYTHONGC:-1}"

# Limit Python's memory allocator
export PYTHONMALLOC="${PYTHONMALLOC:-malloc}"

# -----------------------------------------------------------------------------
# Start server
# -----------------------------------------------------------------------------

log_info "Starting GraphQL AGI server..."
log_info "  Port: $PORT"
log_info "  Workers: $WORKERS"
log_info "  Log Level: $LOG_LEVEL"
log_info "  Database: $DATABASE_PATH"

# Execute with nice priority
exec nice -n "$NICE_LEVEL" python -m uvicorn graphql_agi.server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers "$WORKERS" \
    --limit-concurrency "${CONCURRENCY_LIMIT:-100}" \
    --limit-max-requests "${MAX_REQUESTS:-10000}" \
    --timeout-keep-alive "${KEEPALIVE:-5}" \
    --log-level "$LOG_LEVEL" \
    ${ACCESS_LOG:+--access-log} \
    ${NO_ACCESS_LOG:+--no-access-log}
