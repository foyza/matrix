FROM python:3.11-slim

WORKDIR /app

# Устанавливаем зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Ставим Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Качаем nltk ресурсы (после установки nltk!)
RUN python -m nltk.downloader punkt stopwords

# Копируем весь проект
COPY . .

# Переменные окружения (например для Railway)
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]

