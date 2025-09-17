# Use slim image
FROM python:3.11-slim

# Install system deps needed for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget ca-certificates \
    libgomp1 libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
ENV PIP_NO_CACHE_DIR=off
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY bot.py /app/bot.py

# Run bot
CMD ["python", "bot.py"]
