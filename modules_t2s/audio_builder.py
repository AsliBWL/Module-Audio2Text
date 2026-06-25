# Импорт библиотек
import numpy as np


class AudioBuilder:
    """
    Класс AudioBuilder отвечает за хранение всех синтезированных аудио.
    """

    # Конструктор
    def __init__(self):

        # Создаем список, в котором будут храниться синтезированные аудио
        self.audio_list = []

        # Создаем список, в котором будут храниться частоты дискретизации синтезированных аудио
        self.sample_rate_list = []

    # Функция добавления синтезированного аудио в список
    def add_audio(self, audio: np.ndarray, sample_rate: int):
        """
        Args:
            audio (np.ndarray): Синтезированное аудио в формате float32 (в нормализованном виде).
            sample_rate (int): Частота дискретизации сгенерированного аудио.
        """

        # Добавляем синтезированное аудио в список
        self.audio_list.append(audio)

        # Добавляем частоту дискретизации аудио в список
        self.sample_rate_list.append(sample_rate)
