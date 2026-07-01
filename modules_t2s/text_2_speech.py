# Импорт библиотек
from abc import ABC, abstractmethod
from modules_t2s.config import Config
from qwen_tts import Qwen3TTSModel
import torch
from modules_t2s.audio_converter import AudioConverter


class Text2SpeechInterface(ABC):
    """
    Абстрактный класс Text2SpeechInterface отвечает за синтез речи, т. е. за преобразование текста в аудио.
    """

    # Функция преобразования текста в аудио
    @abstractmethod
    def synthesis(self, text: str):
        """
        Args:
            text(str): Текст, который нужно преобразовать в аудио.
        Returns:
            audio (np.ndarray): Аудиоданные в формате float32 в нормализованном виде.
                                Размер audio (количество семплов, 1).
            sample_rate (int): Частота дискретизации сгенерированного аудио.
        """

        pass


class Qwen3Synthesizer(Text2SpeechInterface):
    """
    Реализация синтеза речи с помощью модели Qwen3-TTS.
    """

    # Конструктор
    def __init__(self, config: Config):
        """
        Args:
            config (Config): Объект класса Config со всеми настройками модуля Text2Speech.
        """

        # Сохраняем конфигурацию для доступа к настройкам модуля Text2Speech
        self.config = config

        # Загружаем модель Qwen3-TTS
        self.model = Qwen3TTSModel.from_pretrained(
            pretrained_model_name_or_path=self.config.qwen_model_name,
            device_map=self.config.qwen_device,
            dtype=self.config.dtype,
            attn_implementation=self.config.attn_implementation
        )

    # Функция преобразования текста в аудио
    def synthesis(self, text: str):
        """
        Args:
            text(str): Текст, который нужно преобразовать в аудио.
        Returns:
            audio (np.ndarray): Аудиоданные в формате float32 в нормализованном виде.
                                Размер audio (количество семплов, 1).
            sample_rate (int): Частота дискретизации сгенерированного аудио.
        """

        # Если была выбрана модель для использования встроенного голоса
        if ((self.config.qwen_model_name == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
                or (self.config.qwen_model_name == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")):

            # Синтезируем аудио
            audio_list, sample_rate = self.model.generate_custom_voice(
                text=text,
                language=self.config.qwen_language,
                speaker=self.config.qwen_speaker,
                instruct=self.config.instruct
            )

        # Если была выбрана модель для клонирования голоса
        elif ((self.config.qwen_model_name == "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
              or (self.config.qwen_model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-Base")):

            # Синтезируем аудио
            audio_list, sample_rate = self.model.generate_voice_clone(
                text=text,
                language=self.config.qwen_language,
                ref_audio=self.config.path_to_audio_for_clone,
                ref_text=self.config.text_for_clone
            )

        # Если была выбрана модель для создания голоса по текстовому описанию
        else:

            # Синтезируем аудио
            audio_list, sample_rate = self.model.generate_voice_design(
                text=text,
                language=self.config.qwen_language,
                instruct=self.config.instruct
            )

        # Получаем синтезированное аудио в формате int16
        audio = audio_list[0]

        # Конвертируем синтезированное аудио в формат float32 (в нормализованный вид)
        audio = AudioConverter.int16_to_float32(audio)

        # Возвращаем синтезированное аудио и его частоту дискретизации
        return audio, sample_rate


class SileroSynthesizer(Text2SpeechInterface):
    """
    Реализация синтеза речи с помощью модели Silero TTS.
    """

    # Конструктор
    def __init__(self, config: Config):
        """
        Args:
            config (Config): Объект класса Config со всеми настройками модуля Text2Speech.
        """

        # Сохраняем конфигурацию для доступа к настройкам модуля Text2Speech
        self.config = config

        # Загружаем модель Silero TTS
        self.model, example_text = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            trust_repo=True,  # чтобы не требовалось разрешение для клонирования репозитория с моделью Silero TTS
            model="silero_tts",
            language=self.config.silero_language,
            speaker=self.config.silero_model_version
        )

        # Переносим модель Silero TTS на устройство для вычислений
        self.model.to(self.config.silero_device)

    # Функция преобразования текста в аудио
    def synthesis(self, text: str):
        """
        Args:
            text(str): Текст, который нужно преобразовать в аудио.
        Returns:
            audio (np.ndarray): Аудиоданные в формате float32 в нормализованном виде.
                                Размер audio (количество семплов, 1).
            sample_rate (int): Частота дискретизации сгенерированного аудио.
        """

        # Синтезируем аудио (аудио получаем в формате float32)
        audio = self.model.apply_tts(
            text=text,
            speaker=self.config.silero_speaker,
            sample_rate=self.config.sample_rate,
            put_accent=self.config.put_accent,
            put_yo=self.config.put_yo
        )

        # Получаем частоту дискретизации синтезированного аудио
        sample_rate = self.config.sample_rate

        # Возвращаем синтезированное аудио и его частоту дискретизации
        return audio, sample_rate
