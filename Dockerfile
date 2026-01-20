# Moon Dev AI Trading Bot - Docker Image
# Optimized for Coolify deployment
# Build: 2026-01-20-v2 - Single API call for all prices (10x faster)

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash trader
RUN mkdir -p /app/src/data/ramf && chown -R trader:trader /app

# Copy requirements first for better caching (use minimal docker requirements)
COPY --chown=trader:trader requirements-docker.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Note: Running as root for volume mount compatibility
# The data volume needs write permissions

# Volume for persistent data (trades, logs, signals)
VOLUME ["/app/src/data"]

# Expose web dashboard port
EXPOSE 8080

# Health check - verify web dashboard is responding (uses public /health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default entry point
ENTRYPOINT ["./entrypoint.sh"]
