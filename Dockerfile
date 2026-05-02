FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY *.py ./
COPY tests/ ./tests/

# Create persistence directory
RUN mkdir -p /data

ENV EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV PERSIST_DIR=/data
ENV PORT=5000

EXPOSE 5000

# Production server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:create_app()"]
