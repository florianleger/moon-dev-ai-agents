# Moon Dev AI Trading Bot - Docker Image
# Optimized for Coolify deployment
# Build: 2026-03-03-v5 - Multi-stage build, non-root user, improved healthcheck

# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies to user site-packages
COPY requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime-only system dependencies (no gcc/g++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash trader

# Create data directories and set permissions BEFORE switching user
RUN mkdir -p /app/src/data/ramf \
    /app/src/data/execution_results \
    /app/src/data/signals \
    /app/data \
    /app/logs \
    && chown -R trader:trader /app

# Copy application code
COPY --chown=trader:trader src/ ./src/
COPY --chown=trader:trader entrypoint.sh .
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER trader

# Volume for persistent data (trades, logs, signals)
VOLUME ["/app/src/data"]

# Expose web dashboard port
EXPOSE 8080

# Health check - verify web dashboard AND main.py heartbeat
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health && \
    python -c "import json,time; d=json.load(open('/app/src/data/bot_heartbeat.json')); assert time.time()-d.get('timestamp',0)<600, 'Bot heartbeat stale'" \
    || exit 1

# Default entry point
ENTRYPOINT ["./entrypoint.sh"]
