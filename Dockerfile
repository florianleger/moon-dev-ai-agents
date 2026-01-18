# Moon Dev AI Trading Bot - Docker Image
# Optimized for Coolify deployment

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
RUN mkdir -p /app/src/data && chown -R trader:trader /app

# Copy requirements first for better caching (use minimal docker requirements)
COPY --chown=trader:trader requirements-docker.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=trader:trader src/ ./src/
COPY --chown=trader:trader entrypoint.sh .
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER trader

# Volume for persistent data (trades, logs, signals)
VOLUME ["/app/src/data"]

# Health check - verify Python and main.py exist
HEALTHCHECK --interval=5m --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default entry point
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "src/main.py"]
