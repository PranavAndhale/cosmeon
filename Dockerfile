FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for rasterio/GDAL
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed logs

# Expose API port
EXPOSE 8000

# Default: run API server
CMD ["python", "main.py", "api"]
