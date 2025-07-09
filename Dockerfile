# Multi-stage build for faster deployment
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install pip and wheel
RUN pip install --upgrade pip setuptools wheel

# Install dependencies using pre-built wheels where possible
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    numpy==1.25.2 \
    scipy==1.11.4 \
    scikit-learn==1.3.2

# Install PyTorch CPU-only (much smaller)
RUN pip install --no-cache-dir torch==2.1.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install ML libraries
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    huggingface_hub==0.19.4 \
    sentence-transformers==2.4.0 \
    faiss-cpu==1.7.4

# Production stage
FROM python:3.11-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application code
COPY ./app /app/app

# Create directories
RUN mkdir -p /app/data/indexes /app/indexes /tmp/indexes /tmp/data

# Expose port
EXPOSE 8000

# Command to run the application  
CMD ["sh", "-c", "uvicorn app.main_working:app --host 0.0.0.0 --port ${PORT:-8000}"]