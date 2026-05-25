# Импорт библиотек
import numpy as np
from config import Config
from audio_converter import AudioConverter


class SessionContext:
    """
    Класс SessionContext отвечает за поддержку состояния сессии распознавания
    ("сейчас говорим/не говорим") и буферизацию аудиоданных.

    Причина создания этого класса:
    Аудиоданные приходят от микрофона фреймами на VAD.
    VAD анализирует их, но не может накапливать речь между фреймами.
    Класс SessionContext служит "буфером" и "памятью":
    - накапливает семплы, пока не наберётся полный фрейм для VAD (512 семплов);
    - хранит, идёт речь или тишина (говорит ли пользователь прямо сейчас);
    - собирает байты речи во время разговора;
    - когда речь закончилась - отдаёт весь накопленный сегмент распознавателю.
    """

    # Конструктор
    def __init__(self, config: Config):
        """
        Args:
            config (Config): Объект класса Config со всеми настройками приложения
        """

        # Сохраняем конфигурацию для доступа к настройкам приложения
        self.config = config

        # Вычисляем размер фрейма в байтах
        self.frame_bytes_len = self.config.vad_frame_samples * self.config.bytes_per_sample

        # Создаем буфер для хранения фреймов в формате float32 в диапазоне -1..1
        """
        Сырые байты от микрофона преобразуем в формат float32 в диапазоне -1..1,
        чтобы VAD мог их обработать
        """
        self.pending_float = np.array([], dtype=np.float32)

        # Создаем буфер для хранения фреймов в формате int16 (сырые байты от микрофона)
        self.pending_bytes = bytearray()

        # Создаем флаг: говорит ли пользователь прямо сейчас
        """
        True - пользователь говорит
        False - тишина
        """
        self.in_speech = False

        # Создаем буфер для накопления текущей фразы (речь между паузами) в байтах в формате int16
        """
        Когда VAD сообщит о конце речи, этот буфер будет преобразован
        в float32 и отправлен в модель распознавания
        """
        self.speech_segment = bytearray()

    # Функция добавления сырого байтового фрейма от микрофона в буфер
    def add_audio_chunk(self, audio_bytes: bytes):
        """
        Args:
            audio_bytes (bytes): Байтовый фрейм от микрофона в формате int16, моно
        """

        # Конвертируем байтовый фрейм от микрофона из int16 в float32
        # и добавляем в конец массива pending_float
        self.pending_float = np.concatenate([
            self.pending_float,
            AudioConverter.pcm16_to_float32(audio_bytes)
        ])

        # Добавляем байтовый фрейм в конец массива pending_bytes
        self.pending_bytes.extend(audio_bytes)

    # Функция проверки: накопилось ли в буфере достаточно данных для одного фрейма (>= 512 семплов)
    def has_complete_frame(self):
        """
        Returns:
            bool: True - данных достаточно, их можно извлекать для отправки на VAD
                  False - нужно подождать ещё данных от микрофона
        """

        # Проверяем оба буфера и возвращаем результат:
        # В float буфере должно быть не меньше 512 семплов
        # В байтовом буфере должно быть не меньше 1024 байт
        return (len(self.pending_float) >= self.config.vad_frame_samples and
                len(self.pending_bytes) >= self.frame_bytes_len)

    # Функция извлечения следующего фрейма из буфера для отправки на VAD
    def pop_frame(self):
        """
        Берёт ровно vad_frame_samples семплов из начала буфера
        и удаляет их из обоих буферов (pending_float и pending_bytes).

        Returns:
            tuple[np.ndarray, bytes]:
                - frame_float: массив float32 для VAD (512 элементов);
                - frame_bytes: байтовый объект с теми же данными (1024 байта).
        """

        # Извлекаем первые vad_frame_samples семплов из буфера pending_float
        frame_float = self.pending_float[:self.config.vad_frame_samples]

        # Обрезаем буфер pending_float: оставляем только элементы после первых vad_frame_samples семплов
        self.pending_float = self.pending_float[self.config.vad_frame_samples:]

        # Извлекаем первые frame_bytes_len байта из буфера pending_bytes
        frame_bytes = bytes(self.pending_bytes[:self.frame_bytes_len])

        # Удаляем из буфера pending_bytes первые frame_bytes_len байта
        del self.pending_bytes[:self.frame_bytes_len]

        # Возвращаем извлечённый фрейм в двух форматах
        return frame_float, frame_bytes

    # Функция начала записи речи
    def start_speech(self):
        """
        Вызывается, когда VAD обнаружил начало речевого сегмента (событие 'start').
        """

        # Устанавливаем значение флага: пользователь говорит
        self.in_speech = True

        # Создаем новый буфер для накопления новой фразы
        self.speech_segment = bytearray()

    # Функция накопления фразы (промежутка аудио между паузами)
    def accumulate_speech(self, frame_bytes: bytes):
        """
        Вызывается для каждого фрейма, когда in_speech == True.
        Добавляет очередной фрейм в буфер speech_segment.

        Args:
            frame_bytes (bytes): Очередной фрейм аудио (1024 байта)
        """

        # Добавляем очередной фрейм в буфер speech_segment
        self.speech_segment.extend(frame_bytes)

    # Функция завершения записи речи
    def end_speech(self):
        """
        Вызывается, когда VAD обнаружил конец речевого сегмента (событие 'end').

        Returns:
            np.ndarray: Массив float32 с аудиоданными всей фразы.
        """

        # Преобразуем накопленные байты в нормализованный массив float32
        """
        1. Байтовый объект в массив int16
        2. Массив int16 в массив float32
        3. Нормализация
        """
        audio_float = np.frombuffer(bytes(self.speech_segment), dtype=np.int16).astype(np.float32) / 32768.0

        # Сбрасываем значение флага: пользователь перестал говорить
        self.in_speech = False

        # Возвращаем нормализованный массив float32 с аудиоданными для распознавания
        return audio_float

    # Функция сброса состояния сессии распознавания
    def reset(self):
        """
        Вызывается, когда нужно полностью очистить текущее состояние:
        - перестать считать, что идёт речь
        - очистить накопленный речевой сегмент

        Отличие от end_speech():
        - end_speech() возвращает аудио и сбрасывает флаг (нормальное завершение)
        - reset() просто сбрасывает, не возвращая аудио (например, при ошибке
          или принудительной остановке)
        """

        # Сбрасываем значение флага
        self.in_speech = False

        # Очищаем буфер с накопленной фразой
        self.speech_segment = bytearray()
