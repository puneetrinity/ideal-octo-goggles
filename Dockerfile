# Fast deployment - minimal ML setup
FROM python:3.11-slim

WORKDIR /app

# Install only essential packages for fast deployment
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0

# Copy application code
COPY ./app /app/app

# Create directories
RUN mkdir -p /app/data/indexes /app/indexes /tmp/indexes /tmp/data

# Expose port
EXPOSE 8000

# Command to run basic FastAPI app without ML dependencies
CMD ["sh", "-c", "uvicorn app.main_basic:app --host 0.0.0.0 --port ${PORT:-8000}"]