# Используем официальный образ Python с slim-версией Debian
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y gcc g++ make && \
    apt-get install -y sox && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements_s2t.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements_s2t.txt

# Копируем файлы приложения
COPY api_t2s.py .
COPY modules/ ./modules/
COPY modules_s2t/ ./modules_s2t/

# Создаем директорию для временных файлов
RUN mkdir -p /tmp/audio_processing

# Определяем volumes для моделей и кэша
VOLUME ["/app/models", "/root/.cache", "/tmp/audio_processing"]

# Определяем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/models

# Открываем порт для API
EXPOSE 8000

# Команда для запуска API по умолчанию
CMD ["uvicorn", "api_t2s:app", "--host", "0.0.0.0", "--port", "8000"]
