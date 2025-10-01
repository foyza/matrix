# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем зависимости системы
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Создаём рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем нужные данные для NLTK
RUN python -m nltk.downloader punkt stopwords

# Копируем код бота
COPY . .

# Запуск
CMD ["python", "main.py"]
