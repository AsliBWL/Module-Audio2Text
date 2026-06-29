# Импорт библиотек
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import io
import numpy as np
import soundfile as sf

# Импорт модулей Speech2Text
from modules_s2t.config import Config as S2TConfig
from modules_s2t.speech_recognizer import FasterWhisperRecognizer


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
    title="Speech to Text API",
    description="API для распознавания речи из аудио с использованием модели Whisper",
    version="1.0.0"
)


# Глобальный конфиг и распознаватель для STT
s2t_config: Optional[S2TConfig] = None
s2t_recognizer: Optional[FasterWhisperRecognizer] = None


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


@app.on_event("startup")
async def startup_event():
    """Инициализирует распознаватель речи при запуске приложения."""
    init_s2t()


@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о API."""
    return {
        "message": "Speech to Text API",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Интерактивная документация Swagger UI",
            "/transcribe_audio": "POST - Распознавание речи из аудио (STT)"
        }
    }


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
