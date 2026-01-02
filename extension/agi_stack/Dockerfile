# =============================================================================
# Ada AGI Surface - Dockerfile
# =============================================================================
# Unified AGI interface: Kuzu + LanceDB + GraphQL + VSA + NARS
# Optimized for Railway deployment with minimal resource usage
# =============================================================================

FROM python:3.11-slim-bookworm

# Labels
LABEL maintainer="Ada AGI Team"
LABEL description="Ada AGI Surface - Unified AGI interface"

# Environment variables for minimal resource usage
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Thread limits (minimize CPU usage)
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    # App settings
    PORT=8080 \
    KUZU_DB_PATH=/data/kuzu \
    LANCE_DB_PATH=/data/lancedb

WORKDIR /app

# Create non-root user
RUN groupadd --gid 1000 agi && \
    useradd --uid 1000 --gid agi --shell /bin/bash --create-home agi

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy requirements first for better caching
COPY extension/ada_surface/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy extension code
COPY extension/ada_surface /app/extension/ada_surface

# Set Python path
ENV PYTHONPATH=/app

# Create data directory (Kuzu will create its own subdirectory)
RUN mkdir -p /data && \
    chown -R agi:agi /app /data

# Switch to non-root user
USER agi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

# Start command
CMD ["python", "-m", "uvicorn", "extension.ada_surface.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--limit-concurrency", "50", "--no-access-log"]
