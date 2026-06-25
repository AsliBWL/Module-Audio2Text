# Импорт библиотек
import numpy as np
import soundfile as sf
from pathlib import Path


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

    # Функция сохранения всех синтезированных аудио в один файл
    def save_merged_audio(self, output_path: str = "merged_audio.wav"):
        """
        Объединяет все синтезированные аудио в один файл и сохраняет его.

        Args:
            output_path (str): Путь для сохранения объединенного аудиофайла.

        Returns:
            str: Путь к сохраненному файлу или None, если нет аудио для сохранения.
        """

        # Проверяем, что есть хотя бы одно синтезированное аудио
        if not self.audio_list:
            return None

        # Объединяем все аудио в один массив
        merged_audio = np.concatenate(self.audio_list)

        # Создаем директорию для сохранения файла, если она не существует
        output_file_path = Path(output_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем объединенное аудио в WAV-файл
        sf.write(str(output_file_path), merged_audio, self.sample_rate_list[0])

        return str(output_file_path)
