# --- Base image ---
FROM python:3.11-slim

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Set workdir ---
WORKDIR /app

# --- Install Python deps ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Download NLTK data (VADER lexicon) ---
RUN python -m nltk.downloader vader_lexicon

# --- Copy project files ---
COPY . .

# --- Env variables ---
ENV PYTHONUNBUFFERED=1

# --- Start bot ---
CMD ["python", "bot.py"]
