FROM nvcr.io/nvidia/nemo:23.10

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    prometheus-node-exporter \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (package manager)
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv --version

# Set working directory
WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY docs/ ./docs/

# Create model cache directory (writable by the container user)
RUN mkdir -p /models

# Expose port
EXPOSE 8000

# Health check (no auth required, uses /healthz endpoint)
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command
CMD ["python", "-m", "src.nemo_openai_server"]
