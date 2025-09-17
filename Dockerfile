# Use slim image
FROM python:3.11-slim

# Install system deps needed for building some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget ca-certificates \
    libatlas-base-dev libgomp1 libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
# Prevent pip from building unnecessary C extensions in parallel (saves memory)
ENV PIP_NO_CACHE_DIR=off
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY bot_project.py /app/bot_project.py
COPY .env.example /app/.env.example

# Expose nothing (bot uses outgoing connections). Use CMD to run.
CMD ["python", "/app/bot_project.py"]
