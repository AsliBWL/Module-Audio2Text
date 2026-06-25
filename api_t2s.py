# Импорт библиотек
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import re
from pathlib import Path
from datetime import datetime
import tempfile
import os

# Импорт модулей Text2Speech
from modules_t2s.context import Context


# Запрос для эндпоинта генерации аудио
class TextToSpeechRequest(BaseModel):
    """
    Запрос для эндпоинта генерации аудио из текста.
    """
    text: str  # Текст для преобразования в аудио
    model_name: Optional[str] = "Silero TTS"  # Модель TTS (Silero TTS или Qwen3-TTS)
    silero_speaker: Optional[str] = "kseniya"  # Голос для Silero TTS
    sample_rate: Optional[int] = 24000  # Частота дискретизации для Silero TTS


# Создаём FastAPI приложение
app = FastAPI(
    title="Text2Speech API",
    description="API для преобразования текста в аудио с использованием моделей Silero TTS и Qwen3-TTS",
    version="1.0.0"
)


# Глобальный контекст для повторного использования модели
context: Optional[Context] = None


def init_context(model_name: str = "Silero TTS", silero_speaker: str = "kseniya", sample_rate: int = 24000):
    """
    Инициализирует контекст для синтеза речи.

    Args:
        model_name: Имя модели TTS
        silero_speaker: Голос для Silero TTS
        sample_rate: Частота дискретизации для Silero TTS
    """
    global context
    if context is None:
        context = Context(
            model_name=model_name,
            silero_speaker=silero_speaker,
            sample_rate=sample_rate
        )


def split_into_sentences(text: str, sentences_per_chunk: int = 2):
    """
    Разбивает текст на группы по заданному количеству предложений.

    Args:
        text: Исходный текст
        sentences_per_chunk: Количество предложений в одной группе

    Returns:
        list: Список групп предложений
    """
    # Разбиваем текст на предложения, сохраняя разделители
    sentences = re.split(r'([.!?]+\s+)', text)

    # Объединяем разделители с предложениями
    full_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
            if sentence.strip():
                full_sentences.append(sentence.strip())

    # Если есть последнее предложение без разделителя
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        full_sentences.append(sentences[-1].strip())

    # Группируем предложения по chunks
    chunks = []
    for i in range(0, len(full_sentences), sentences_per_chunk):
        chunk = ' '.join(full_sentences[i:i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks if chunks else [text]


@app.on_event("startup")
async def startup_event():
    """Инициализирует контекст при запуске приложения."""
    init_context()


@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о API."""
    return {
        "message": "Text2Speech API",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Интерактивная документация Swagger UI",
            "/generate_speech": "POST - Генерация аудио из текста"
        }
    }


@app.post("/generate_speech")
async def generate_speech(request: TextToSpeechRequest):
    """
    Генерирует аудио из текста с разбивкой на группы по 2 предложения.

    Args:
        request: Запрос с текстом и параметрами модели

    Returns:
        FileResponse: Аудиофайл в формате WAV
    """

    try:
        # Проверяем, что текст не пустой
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Текст не может быть пустым")

        # Инициализируем или обновляем контекст при необходимости
        global context
        if context is None or (
            hasattr(context, 'config') and
            context.config.silero_speaker != request.silero_speaker or
            context.config.sample_rate != request.sample_rate
        ):
            context = Context(
                model_name=request.model_name,
                silero_speaker=request.silero_speaker,
                sample_rate=request.sample_rate
            )

        # Разбиваем текст на группы по 2 предложения
        text_chunks = split_into_sentences(request.text, sentences_per_chunk=2)

        # Генерируем аудио для каждой группы
        for i, chunk in enumerate(text_chunks, 1):
            print(f"Генерация аудио для группы {i}/{len(text_chunks)}: '{chunk[:50]}...'")
            context.start(chunk)

        # Создаём временную директорию для сохранения аудио
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"speech_{timestamp}.wav"
        output_path = os.path.join(temp_dir, output_filename)

        # Сохраняем объединённое аудио
        saved_path = context.save_audio(output_path)

        if not saved_path:
            raise HTTPException(status_code=500, detail="Не удалось сохранить аудио")

        # Возвращаем файл пользователю
        return FileResponse(
            path=saved_path,
            media_type="audio/wav",
            filename=output_filename,
            background=lambda: cleanup_temp_file(temp_dir)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации аудио: {str(e)}")


def cleanup_temp_file(temp_dir: str):
    """
    Удаляет временную директорию после отправки файла пользователю.

    Args:
        temp_dir: Путь к временной директории
    """
    try:
        if os.path.exists(temp_dir):
            os.remove(os.path.join(temp_dir, os.listdir(temp_dir)[0]))
            os.rmdir(temp_dir)
    except Exception:
        pass  # Игнорируем ошибки при очистке


if __name__ == "__main__":
    import uvicorn

    # Запуск сервера
    uvicorn.run(
        "api_t2s:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
