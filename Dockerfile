# Use a pre-built image with ML dependencies
FROM continuumio/miniconda3:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages quickly with conda
RUN conda install -y -c conda-forge \
    python=3.11 \
    numpy \
    scipy \
    scikit-learn \
    && conda clean -ya

# Install remaining packages with pip (much faster now)
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers==4.36.2 \
    huggingface_hub==0.19.4 \
    sentence-transformers==2.4.0 \
    faiss-cpu==1.7.4

WORKDIR /app

# Copy application code
COPY ./app /app/app

# Create directories
RUN mkdir -p /app/data/indexes /app/indexes /tmp/indexes /tmp/data

# Expose port
EXPOSE 8000

# Command to run the application  
CMD ["sh", "-c", "uvicorn app.main_working:app --host 0.0.0.0 --port ${PORT:-8000}"]