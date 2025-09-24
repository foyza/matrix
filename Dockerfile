# Базовый образ с Python
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем ресурсы для nltk (токенизаторы, стоп-слова и т.д.)
RUN python -m nltk.downloader punkt stopwords

# Копируем всё приложение
COPY . .

# Переменные окружения (лучше пробрасывать через .env или Railway Secrets)
ENV PYTHONUNBUFFERED=1

# Запуск бота
CMD ["python", "main.py"]
