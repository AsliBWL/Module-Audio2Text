# Импорт библиотек
import numpy as np


class AudioConverter:
    """
    Утилитарный класс AudioConverter отвечает за конвертацию синтезированного аудио.
    """

    # Функция конвертации int16 в float32 в диапазоне -1..1
    @staticmethod  # Метод класса будет статическим
    def int16_to_float32(audio: np.ndarray):
        """
        Args:
            audio (np.ndarray): Синтезированные аудиоданные в формате int16.
                                Размер audio (количество семплов, 1).

        Returns:
            audio_float32 (np.ndarray): Синтезированные аудиоданные в формате float32,
                                        нормализованные в диапазоне -1..1.
                                        Размер audio (количество семплов, 1).
        """

        # Преобразуем массив int16 в массив float32
        float32_array = audio.astype(np.float32)

        # Нормализуем массив float32 в диапазон -1..1
        audio_float32 = float32_array / 32768.0

        # Возвращаем массив с аудиоданными в формате float32, нормализованными в диапазоне -1..1
        return audio_float32
