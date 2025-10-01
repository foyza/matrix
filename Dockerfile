# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем зависимости системы (для numpy, pandas, ta, sklearn, tensorflow)
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем requirements и ставим зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Указываем переменные окружения (чтоб .env подтянулся внутри Railway/докера)
ENV PYTHONUNBUFFERED=1

# Запуск бота
CMD ["python", "main.py"]

