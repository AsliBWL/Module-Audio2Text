# Импорт библиотек
from abc import ABC, abstractmethod
import numpy as np
from faster_whisper import WhisperModel
from modules_s2t.config import Config

# --------------------------------------------------------------------------------------------
# ------------------------------------ НАЧАЛО НОВОГО КОДА ------------------------------------
# --------------------------------------------------------------------------------------------
import asyncio
import websockets
import json
import base64
import soundfile as sf
import io
import numpy as np
from modules_s2t.config import Config
# --------------------------------------------------------------------------------------------
# ------------------------------------- КОНЕЦ НОВОГО КОДА ------------------------------------
# --------------------------------------------------------------------------------------------


class SpeechRecognizerInterface(ABC):
    """
    Абстрактный класс SpeechRecognizerInterface отвечает за
    распознавание речи, т. е. за преобразование аудио в текст.
    """

    # Функция преобразования аудио в текст
    """
    Метод класса будет абстрактным, т. е. не имеет тела,
    но обязан быть реализован в классе наследнике
    """
    @abstractmethod
    def recognize(self, audio: np.ndarray):
        """
        Args:
            audio (np.ndarray): Аудио в формате float32.
                                Может быть любой длины: от короткой команды до часовой записи.

        Returns:
            tuple[str, str, float]: Кортеж из трёх элементов:
                                    - text (str): Распознанный текст;
                                    - language (str): Код языка, на котором говорили (например, 'ru', 'en', 'fr');
                                    - language_prob (float): Уверенность модели в определении языка (0.0 - 1.0).
        """

        pass


class FasterWhisperRecognizer(SpeechRecognizerInterface):
    """
    Реализация распознавания речи через библиотеку faster-whisper.
    """

    # Конструктор
    def __init__(self, config: Config):
        """
        Args:
            config (Config): Объект класса Config со всеми настройками приложения
        """

        # Сохраняем конфигурацию для доступа к настройкам приложения
        self.config = config

        # Загружаем модель faster-whisper
        self.model = WhisperModel(
            config.whisper_model_version,      # версия модели Whisper
            device=config.device,              # устройство для вычислений
            compute_type=config.compute_type,  # тип вычислений
        )

    # Функция преобразования аудио в текст
    def recognize(self, audio: np.ndarray):
        """
        Args:
            audio (np.ndarray): Аудио в формате float32.
                                Может быть любой длины: от короткой команды до часовой записи.

        Returns:
            tuple[str, str, float]: Кортеж из трёх элементов:
                                    - text (str): Распознанный текст;
                                    - language (str): Код языка, на котором говорили (например, 'ru', 'en', 'fr');
                                    - language_prob (float): Уверенность модели в определении языка (0.0 - 1.0).
        """

        # Вызываем транскрипцию аудио
        """
        Метод transcribe() возвращает генератор segments (сегментов текста)
        и объект info с информацией о транскрипции.
        """
        segments, info = self.model.transcribe(
            audio,                                                                 # numpy-массив с аудиоданными
            beam_size=self.config.beam_size,                                       # количество наиболее вероятных
                                                                                   # вариантов расшифровки аудио,
                                                                                   # рассматриваемых параллельно
            language=None if self.config.auto_language else self.config.language,  # язык (None для автоопределения)
            vad_filter=False,                                                      # использовать ли встроенный VAD
                                                                                   # (VAD уже используется отдельно)
        )

        # Объединяем текст всех сегментов в одну строку через пробел
        """
        segments - это генератор, который выдает объекты Segment по одному.
        
        Каждый Segment содержит:
        - text: распознанный текст для этого сегмента;
        - start: время начала сегмента (в секундах);
        - end: время конца сегмента (в секундах);
        - avg_logprob: средняя логарифмическая вероятность.
        
        Пример сегмента: Segment(text="Привет мир", start=0.0, end=2.5, avg_logprob=-0.25)
        """
        text = " ".join(segment.text for segment in segments)

        # Извлекаем информацию о языке и уверенности из объекта info
        """
        getattr(object, attribute_name, default_value) - безопасное получение атрибута.
        
        Если у объекта info нет атрибута 'language', вернём 'unknown'.
        Это нужно для совместимости с разными версиями faster-whisper.
        
        Что содержит info (обычно):
        - language: код языка ('ru', 'en', 'fr' и т.д.);
        - language_probability: уверенность определения языка (0.0 - 1.0);
        - duration: длительность аудио в секундах;
        - transcription_time: время, затраченное на транскрипцию.
        """
        language = getattr(info, 'language', 'unknown')
        language_prob = getattr(info, 'language_probability', 0.0)

        print(type(self.model))

        # Возвращаем результат в виде кортежа (текст, язык, уверенность)
        return text, language, language_prob


# --------------------------------------------------------------------------------------------
# ------------------------------------ НАЧАЛО НОВОГО КОДА ------------------------------------
# --------------------------------------------------------------------------------------------
class RemoteWhisperRecognizer(SpeechRecognizerInterface):
    """
    Распознавание речи через удалённый сервер whisper-live.
    Отправляет аудио по WebSocket и получает текст.
    """

    def __init__(self, config: Config, server_host: str = "192.168.1.100"):
        """
        Args:
            config: объект конфигурации (нужен для параметров языка и т.д.)
            server_host: IP-адрес ПК с GPU, где запущен whisper-live
        """
        self.config = config
        self.server_host = server_host
        self.websocket_url = f"ws://{server_host}:9090"

    def recognize(self, audio: np.ndarray) -> tuple[str, str, float]:
        """
        Отправляет аудио на сервер и возвращает результат.

        Args:
            audio: массив float32 с частотой 16000 Гц (уже нормализован)

        Returns:
            (text, language, confidence)
        """
        # Конвертируем float32 в int16 (как ожидает сервер)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Сохраняем во временный буфер в формате WAV
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, 16000, format='WAV')
        wav_bytes = buffer.getvalue()

        # Отправляем через WebSocket и получаем результат
        try:
            # Для синхронного вызова используем asyncio.run()
            text = asyncio.run(self._send_audio_and_receive(wav_bytes))
            # Сервер возвращает только текст, язык и уверенность не передаются
            return text, self.config.language, 1.0
        except Exception as e:
            print(f"❌ Ошибка при обращении к удалённому серверу: {e}")
            return "", self.config.language, 0.0

    async def _send_audio_and_receive(self, wav_bytes: bytes) -> str:
        """
        Асинхронная отправка аудио и получение результата.
        """
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                # Отправляем аудио
                await websocket.send(wav_bytes)
                # Получаем результат (сервер может вернуть JSON или plain text)
                response = await websocket.recv()

                # Пытаемся распарсить JSON
                try:
                    data = json.loads(response)
                    return data.get("text", "")
                except json.JSONDecodeError:
                    return response
        except asyncio.TimeoutError:
            return "Таймаут подключения к серверу распознавания"
        except ConnectionRefusedError:
            return "Сервер распознавания недоступен"
# --------------------------------------------------------------------------------------------
# ------------------------------------- КОНЕЦ НОВОГО КОДА ------------------------------------
# --------------------------------------------------------------------------------------------
