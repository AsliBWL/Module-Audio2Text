# Импорт библиотек
from abc import ABC, abstractmethod
import numpy as np
from faster_whisper import WhisperModel
from config import Config


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
