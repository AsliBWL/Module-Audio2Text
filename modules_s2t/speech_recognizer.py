# Импорт библиотек
from abc import ABC, abstractmethod
import numpy as np
from faster_whisper import WhisperModel
from modules_s2t.config import Config
import asyncio
import websockets
import json
import soundfile as sf
import io


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

        # Возвращаем результат в виде кортежа (текст, язык, уверенность)
        return text, language, language_prob


class RemoteWhisperRecognizer(SpeechRecognizerInterface):
    """
    Реализация распознавания речи через удалённый сервер whisper-live.
    Отправляет аудиоданные по протоколу WebSocket на удалённый сервер whisper-live
    и получает от него распознанный текст.
    """

    # Конструктор
    def __init__(self, config: Config):
        """
        Args:
            config (Config): Объект класса Config со всеми настройками приложения
        """

        # Сохраняем конфигурацию для доступа к настройкам приложения
        self.config = config

        # Инициализируем WebSocket-адрес удалённого сервера whisper-live
        self.websocket_url = f"ws://{self.config.server_host}:9090"

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

        # Конвертируем аудиоданные из float32 в int16 (так ожидает удалённый сервер)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Сохраняем аудиоданные во временный буфер в формате WAV
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, self.config.sample_rate, format='WAV')
        wav_bytes = buffer.getvalue()

        # Пытаемся
        try:
            # отправить аудиоданные по по протоколу WebSocket на удаленный сервер и получить от него результат
            """
            asyncio.run() - функция, которая позволяет вызвать асинхронную функцию оттуда, где асинхронность
            не поддерживается (например, из обычной синхронной функции или из основного потока программы).
            """
            text = asyncio.run(self._send_audio_and_receive(wav_bytes))
            # Возвращаем текст, полученный от сервера, а также язык и уверенность
            """
            Сервер не возвращает язык и уверенность, поэтому задаем их сами.
            """
            return text, self.config.language, 1.0
        # Если возникла ошибка
        except Exception as e:
            # print(f"❌ Ошибка при обращении к удалённому серверу: {e}")
            # Возвращаем пустую строку, а также язык и уверенность
            return "", self.config.language, 0.0

    # Функция отправки аудиоданных на удалённый сервер whisper-live и получения от него результатов
    async def _send_audio_and_receive(self, wav_bytes: bytes):
        """
        Ards:
            wav_bytes (bytes): Аудиоданные в байтовом формате, упакованные в WAV-файл.
        Returns:
            str: Распознанный моделью Whisper текст.

        async - означает, что функция асинхронная (корутина) - специальный тип функций в Python,
        которые могут приостанавливать своё выполнение, не блокируя всю программу, и ждать завершения
        каких-либо операций (например, ответа от сервера или чтения файла).

        await - означает «подожди, пока эта задача не будет выполнена, но пока ты ждёшь,
        занимайся чем-нибудь другим, не блокируй всю программу».
        """

        # Пытаемся
        try:
            # установить WebSocket-соединение с удаленным сервером
            async with websockets.connect(self.websocket_url) as websocket:
                # Отправляем аудио на удаленный сервер
                await websocket.send(wav_bytes)
                # Получаем результат от удаленного сервера (сервер может вернуть JSON или plain text)
                response = await websocket.recv()
                # Пытаемся
                try:
                    # распарсить JSON
                    data = json.loads(response)
                    # Если парсинг прошёл успешно, ищем в полученном словаре значение по ключу "text"
                    # и возвращаем его. Если такого ключа нет, возвращает пустую строку
                    return data.get("text", "")
                # Если распарсить не удалось (сервер вернул plain text)
                except json.JSONDecodeError:
                    # Возвращаем строку, которую вернул сервер
                    return response
        # Если сервер принял соединение, но не отвечает в течение таймаута
        except asyncio.TimeoutError:
            return "Таймаут подключения к серверу распознавания"
        # Если сервер не отвечает на попытку подключения
        except ConnectionRefusedError:
            return "Сервер распознавания недоступен"
