FROM python:3.9-slim

# Установка системных зависимостей для MT5
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Установка Wine для MetaTrader (если нужно эмулировать Windows)
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y wine wine32

# Создание рабочей директории
WORKDIR /app

# Копирование requirements и установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание директорий для логов и данных
RUN mkdir -p /app/logs /app/data

# Переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Запуск бота
CMD ["python", "src/main.py"]
