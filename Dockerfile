# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /test

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/test

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY test/ ./test/
COPY models/ ./models/
COPY esp_vibration_analysis.py .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash test && \
    chown -R test:test /test
USER test

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "test.app:app", "--host", "0.0.0.0", "--port", "8000"]