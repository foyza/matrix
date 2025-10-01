#FROM python:3.11-slim

# Устанавливаем системные зависимости (BLAS, LAPACK и компиляторы)
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем requirements и ставим зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
