FROM python:3.11-slim

# Устанавливаем системные зависимости для numpy/pandas/scikit-learn/tensorflow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем зависимости Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]

