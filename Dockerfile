# Используем официальный образ Python с slim-версией Debian
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libsndfile1 \
    PortAudio \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements_s2t.txt .
COPY requirements_t2s.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements_s2t.txt
RUN pip install --no-cache-dir -r requirements_t2s.txt

# Копируем файлы приложения
COPY api_t2s.py .
COPY main_s2t.py .
COPY modules/ ./modules/
COPY modules_s2t/ ./modules_s2t/
COPY modules_t2s/ ./modules_t2s/

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
