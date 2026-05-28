# Multi-stage Dockerfile для CPU-only режима
FROM python:3.11-slim as builder

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем зависимости в правильном порядке
RUN pip install --no-cache-dir --user \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Stage 2: Final
FROM python:3.11-slim

WORKDIR /app

# Установка runtime зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости из builder
COPY --from=builder /root/.local /root/.local

# Копируем код приложения
COPY ./app ./app
COPY ./data ./data
COPY ./chroma_db ./chroma_db

# Копируем конфиги
COPY .env .env

# Обновляем PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Отключаем CUDA/GPU явно
ENV CUDA_VISIBLE_DEVICES=-1
ENV TORCH_DEVICE=cpu
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_OFFLINE=0

# Создаем директории
RUN mkdir -p /app/logs /app/metrics

# Открываем порт
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Запуск приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]