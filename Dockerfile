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

# Скачать данные для nltk (например, punkt для токенизации)
RUN python -m nltk.downloader punkt stopwords

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё приложение
COPY . .

# Указываем переменные окружения (лучше .env пробросить через Railway/другой деплой)
ENV PYTHONUNBUFFERED=1

# Запуск бота
CMD ["python", "main.py"]
