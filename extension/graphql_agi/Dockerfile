# =============================================================================
# GraphQL AGI - Optimized Dockerfile for Railway
# =============================================================================
# Optimized for:
# - Minimal CPU usage (no GPU, O(1) operations)
# - Small image size (multi-stage build)
# - Fast startup (precomputed indexes)
# - Low memory footprint
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build the C++ extension
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS cpp-builder

# Install build dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /build

# Copy only what's needed for C++ build
COPY CMakeLists.txt ./
COPY src/ ./src/
COPY extension/graphql_agi/ ./extension/graphql_agi/
COPY third_party/ ./third_party/

# Build with optimizations for size and CPU efficiency
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O2 -march=x86-64 -mtune=generic -ffunction-sections -fdata-sections" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--gc-sections" \
    -DBUILD_EXTENSIONS=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARK=OFF \
    && cmake --build build --target graphql_agi -j2

# -----------------------------------------------------------------------------
# Stage 2: Build Python package
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS python-builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir build wheel

# Copy Python package
COPY extension/graphql_agi/python/ ./

# Build wheel
RUN python -m build --wheel --no-isolation

# -----------------------------------------------------------------------------
# Stage 3: Production image (minimal)
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS production

# Labels
LABEL maintainer="GraphQL AGI Team"
LABEL description="GraphQL AGI - AI-native graph database interface"
LABEL version="1.0.0"

# Environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Limit Python's GC aggressiveness
    PYTHONGC=1 \
    # Limit thread usage
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    # GraphQL AGI specific
    GRAPHQL_AGI_MAX_WORKERS=2 \
    GRAPHQL_AGI_CACHE_SIZE=1000 \
    GRAPHQL_AGI_PRECOMPUTE_ON_START=true \
    # Uvicorn settings for low CPU
    WEB_CONCURRENCY=1 \
    UVICORN_WORKERS=1

# Create non-root user for security
RUN groupadd --gid 1000 agi && \
    useradd --uid 1000 --gid agi --shell /bin/bash --create-home agi

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy built extension from cpp-builder
COPY --from=cpp-builder /build/build/extension/graphql_agi/*.so /app/lib/

# Copy and install Python wheel
COPY --from=python-builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Install minimal production dependencies
RUN pip install --no-cache-dir \
    uvicorn[standard]==0.24.0 \
    gunicorn==21.2.0 \
    && pip cache purge

# Copy application code
COPY extension/graphql_agi/python/graphql_agi/ /app/graphql_agi/

# Copy configuration files
COPY extension/graphql_agi/docker/config/ /app/config/

# Create directories for data with proper permissions
RUN mkdir -p /app/data /app/cache /app/logs && \
    chown -R agi:agi /app

# Switch to non-root user
USER agi

# Health check (lightweight)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command with resource-conscious settings
CMD ["python", "-m", "uvicorn", "graphql_agi.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--limit-concurrency", "100", \
     "--limit-max-requests", "10000", \
     "--timeout-keep-alive", "5", \
     "--no-access-log"]
