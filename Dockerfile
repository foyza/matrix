# Используем лёгкий Python-образ
FROM python:3.11-slim

# Устанавливаем зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Переменные окружения (подхватит Railway)
ENV PYTHONUNBUFFERED=1

# Запуск бота
CMD ["python", "main.py"]
