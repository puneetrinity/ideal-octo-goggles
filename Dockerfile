# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for numpy, scipy, and faiss
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies in stages to avoid memory issues
RUN pip install --no-cache-dir numpy==1.25.2
RUN pip install --no-cache-dir scipy
RUN pip install --no-cache-dir scikit-learn==1.3.2
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers==4.36.2
RUN pip install --no-cache-dir huggingface_hub==0.19.4
RUN pip install --no-cache-dir sentence-transformers==2.4.0
RUN pip install --no-cache-dir faiss-cpu==1.7.4
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY ./app /app/app

# Create directories for local development (Railway volume will override /app/data)
RUN mkdir -p /app/data/indexes /app/indexes

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]