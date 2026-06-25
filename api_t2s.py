# Импорт библиотек
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import re
from pathlib import Path
from datetime import datetime
import tempfile
import os
import io
import numpy as np
import soundfile as sf

# Импорт модулей Text2Speech
from modules_t2s.context import Context as T2SContext

# Импорт модулей Speech2Text
from modules_s2t.config import Config as S2TConfig
from modules_s2t.speech_recognizer import FasterWhisperRecognizer


# Запрос для эндпоинта генерации аудио
class TextToSpeechRequest(BaseModel):
    """
    Запрос для эндпоинта генерации аудио из текста.
    """
    text: str  # Текст для преобразования в аудио
    model_name: Optional[str] = "Silero TTS"  # Модель TTS (Silero TTS или Qwen3-TTS)
    silero_speaker: Optional[str] = "kseniya"  # Голос для Silero TTS
    sample_rate: Optional[int] = 24000  # Частота дискретизации для Silero TTS


# Запрос для эндпоинта распознавания речи
class SpeechToTextRequest(BaseModel):
    """
    Запрос для эндпоинта распознавания речи из аудио.
    """
    whisper_model_version: Optional[str] = "small"  # Версия модели Whisper
    device: Optional[str] = "cpu"  # Устройство для вычислений
    language: Optional[str] = "ru"  # Язык распознавания
    auto_language: Optional[bool] = False  # Автоопределение языка


# Создаём FastAPI приложение
app = FastAPI(
    title="Audio Processing API",
    description="API для преобразования текста в аудио (TTS) и аудио в текст (STT) с использованием моделей Silero TTS, Qwen3-TTS и Whisper",
    version="1.0.0"
)


# Глобальный контекст для повторного использования модели TTS
tts_context: Optional[T2SContext] = None

# Глобальный конфиг и распознаватель для STT
s2t_config: Optional[S2TConfig] = None
s2t_recognizer: Optional[FasterWhisperRecognizer] = None


def init_context(model_name: str = "Silero TTS", silero_speaker: str = "kseniya", sample_rate: int = 24000):
    """
    Инициализирует контекст для синтеза речи.

    Args:
        model_name: Имя модели TTS
        silero_speaker: Голос для Silero TTS
        sample_rate: Частота дискретизации для Silero TTS
    """
    global tts_context
    if tts_context is None:
        tts_context = T2SContext(
            model_name=model_name,
            silero_speaker=silero_speaker,
            sample_rate=sample_rate
        )


def init_s2t(whisper_model_version: str = "small", device: str = "cpu",
             language: str = "ru", auto_language: bool = False):
    """
    Инициализирует распознаватель речи.

    Args:
        whisper_model_version: Версия модели Whisper
        device: Устройство для вычислений
        language: Язык распознавания
        auto_language: Автоопределение языка
    """
    global s2t_config, s2t_recognizer
    if s2t_config is None or s2t_recognizer is None:
        s2t_config = S2TConfig(
            whisper_model_version=whisper_model_version,
            device=device,
            language=language,
            auto_language=auto_language
        )
        s2t_recognizer = FasterWhisperRecognizer(s2t_config)


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
    """Инициализирует контексты при запуске приложения."""
    init_context()
    init_s2t()


@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о API."""
    return {
        "message": "Audio Processing API",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Интерактивная документация Swagger UI",
            "/generate_speech": "POST - Генерация аудио из текста (TTS)",
            "/transcribe_audio": "POST - Распознавание речи из аудио (STT)"
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
        global tts_context
        if tts_context is None or (
            hasattr(tts_context, 'config') and
            tts_context.config.silero_speaker != request.silero_speaker or
            tts_context.config.sample_rate != request.sample_rate
        ):
            tts_context = T2SContext(
                model_name=request.model_name,
                silero_speaker=request.silero_speaker,
                sample_rate=request.sample_rate
            )

        # Разбиваем текст на группы по 2 предложения
        text_chunks = split_into_sentences(request.text, sentences_per_chunk=2)

        # Генерируем аудио для каждой группы
        for i, chunk in enumerate(text_chunks, 1):
            print(f"Генерация аудио для группы {i}/{len(text_chunks)}: '{chunk[:50]}...'")
            tts_context.start(chunk)

        # Создаём временную директорию для сохранения аудио
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"speech_{timestamp}.wav"
        output_path = os.path.join(temp_dir, output_filename)

        # Сохраняем объединённое аудио
        saved_path = tts_context.save_audio(output_path)

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


@app.post("/transcribe_audio")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    whisper_model_version: Optional[str] = "small",
    device: Optional[str] = "cpu",
    language: Optional[str] = "ru",
    auto_language: Optional[bool] = False
):
    """
    Распознает речь из аудиофайла и возвращает текст.

    Args:
        audio_file: Загруженный аудиофайл (WAV, MP3 и другие форматы)
        whisper_model_version: Версия модели Whisper
        device: Устройство для вычислений
        language: Язык распознавания
        auto_language: Автоопределение языка

    Returns:
        JSONResponse: JSON с распознанным текстом и информацией о языке
    """

    try:
        # Проверяем, что файл загружен
        if not audio_file or not audio_file.filename:
            raise HTTPException(status_code=400, detail="Файл не загружен")

        print(f"Загружен файл: {audio_file.filename}, тип: {audio_file.content_type}")

        # Инициализируем или обновляем распознаватель при необходимости
        global s2t_config, s2t_recognizer
        if s2t_config is None or s2t_recognizer is None or \
           s2t_config.whisper_model_version != whisper_model_version or \
           s2t_config.device != device or \
           s2t_config.language != language or \
           s2t_config.auto_language != auto_language:

            s2t_config = S2TConfig(
                whisper_model_version=whisper_model_version,
                device=device,
                language=language,
                auto_language=auto_language
            )
            s2t_recognizer = FasterWhisperRecognizer(s2t_config)
            print(f"Распознаватель инициализирован: model={whisper_model_version}, device={device}")

        # Читаем загруженный файл
        audio_content = await audio_file.read()

        # Загружаем аудиофайл с помощью soundfile
        try:
            audio_array, sample_rate = sf.read(io.BytesIO(audio_content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Не удалось прочитать аудиофайл: {str(e)}")

        # Если аудио стерео, конвертируем в моно
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        # Ресемплируем до 16kHz если нужно (Whisper требует 16kHz)
        if sample_rate != 16000:
            # Простое ресемплирование через NumPy (для точности лучше использовать librosa)
            from scipy import signal
            number_of_samples = round(len(audio_array) * float(16000) / sample_rate)
            audio_array = signal.resample(audio_array, number_of_samples)

        # Нормализуем аудио в float32 диапазоне -1..1
        audio_array = audio_array.astype(np.float32)
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()

        print(f"Аудио загружено: длина={len(audio_array)}, sample_rate={sample_rate}")

        # Распознаем речь
        text, detected_lang, lang_prob = s2t_recognizer.recognize(audio_array)

        print(f"Распознанный текст: '{text}', язык: {detected_lang}, уверенность: {lang_prob:.2f}")

        # Возвращаем результат в формате JSON
        return JSONResponse({
            "text": text.strip(),
            "language": detected_lang,
            "language_probability": float(lang_prob),
            "audio_filename": audio_file.filename,
            "audio_duration_seconds": len(audio_array) / 16000.0
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при распознавании речи: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Запуск сервера
    uvicorn.run(
        "api_t2s:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
