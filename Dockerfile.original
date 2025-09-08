# Multi-stage build for optimized image size
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY backend/ ./backend/

# Set environment variables for memory optimization
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
ENV TOKENIZERS_PARALLELISM=false

# Enable extreme memory optimization
ENV USE_EXTREME_OPTIMIZATION=true
ENV MAX_MEMORY_GB=3

WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start the application with memory-limited configuration
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]