# Импорт библиотек
from modules_t2s.config import Config
from modules_t2s.text_2_speech import Qwen3Synthesizer, SileroSynthesizer
from modules_t2s.audio_player import AudioPlayer
from modules_t2s.audio_builder import AudioBuilder
import torch


class Context:
    """
    Класс Context объединяет в себе все объекты, необходимые для синтеза аудио:
    - объект класса Config: хранение значений параметров настройки модуля Text2Speech;
    - объект класса SileroSynthesizer: синтез аудио;
    - объект класса AudioPlayer: воспроизведение синтезированного аудио;
    - объект класса AudioBuilder: хранение синтезированных аудиоданных.
    """

    # Конструктор
    def __init__(self, model_name: str = "Silero TTS",
                 qwen_model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                 qwen_language: str = "Russian", qwen_speaker: str = "Aiden",
                 path_to_audio_for_clone: str = "", text_for_clone: str = "",
                 instruct: str = "", qwen_device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 attn_implementation: str = "flash_attention_2",
                 silero_model_version: str = "v5_5_ru", silero_language: str = "ru",
                 silero_speaker: str = "kseniya", sample_rate: int = 24000,
                 put_accent: bool = True, put_yo: bool = True,
                 silero_device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Args:
            model_name (str): Имя выбранной модели Text2Speech:
                              - "Qwen3-TTS";
                              - "Silero TTS".
            qwen_model_name (str): Имя модели Qwen3-TTS:
                                   - "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";
                                   - "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";
                                   - "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
                                   - "Qwen/Qwen3-TTS-12Hz-1.7B-Base";
                                   - "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign".
                                   Информацию о функциях указанных выше моделей см. в файле config.py (modules_t2s).
            qwen_language (str): Язык генерации речи для модели Qwen3-TTS:
                                 - Русский - "Russian";
                                 - Китайский - "Chinese";
                                 - Английский - "English";
                                 - Немецкий - "German";
                                 - Французский - "French";
                                 - Итальянский - "Italian";
                                 - Японский - "Japanese";
                                 - Корейский - "Korean";
                                 - Португальский - "Portuguese";
                                 - Испанский - "Spanish";
                                 - Автоопределение - "auto".
            qwen_speaker (str): Имя встроенного голоса для модели Qwen3-TTS:
                                - "Aiden" - английский (мужской, звонкий американский голос);
                                - "Dylan" - китайский (мужской, пекинский диалект);
                                - "Eric" - китайский (мужской, сычуаньский диалект);
                                - "Ono_Anna" - японский (женский, игривый голос);
                                - "Ryan" - английский (мужской, динамичный голос);
                                - "Serena" - китайский (женский, теплый и мягкий голос);
                                - "Sohee" - корейский (женский, теплый голос);
                                - "Uncle_Fu" - китайский (мужской, голос солидного возраста);
                                - "Vivian" - китайский (женский, яркий голос).
                                НО все голоса могут синтезировать речь на любом из 10 поддерживаемых моделью языков.
            path_to_audio_for_clone (str): Путь к аудиофайлу с голосом, который нужно клонировать, для модели Qwen3-TTS.
            text_for_clone (str): Текст, который произносится в аудиофайле path_to_audio_for_clone.
            instruct (str): Инструкции для управления тоном и эмоциями встроенного голоса или
                            для создания нового голоса по текстовому описанию для модели Qwen3-TTS.
                            Пример: "Speak happily and joyfully, with a cheerful tone and a smile in your voice."
            qwen_device (str): Устройство для вычислений, выполняемых моделью Qwen3-TTS:
                               - "cuda:n" - GPU №n;
                               - "cuda" - GPU (автовыбор);
                               - "cpu" - CPU;
                               - "auto" - автовыбор устройства.
            dtype (torch.dtype): Тип данных для весов модели Qwen3-TTS:
                                 - torch.float16;
                                 - torch.bfloat16;
                                 - torch.float32.
                                 Информацию о указанных выше типах данных см. в файле config.py (modules_t2s).
            attn_implementation (str): Реализация механизма внимания для модели Qwen3-TTS:
                                       - None - стандартная реализация PyTorch;
                                       - "flash_attention_2" - алгоритм Flash Attention версия 2 (стабильная версия);
                                       - "sdpa" - встроенная в PyTorch оптимизация;
                                       - "kernels-community/flash-attn3" - алгоритм Flash Attention версия 3
                                       (экспериментальная версия).
                                       Подробнее о Flash Attention см. в файле config.py (modules_t2s).
            silero_model_version (str): Версия модели Silero TTS:
                                        - "v5_5_ru";
                                        - "v5_4_ru";
                                        - "v5_3_ru";
                                        - "v5_2_ru";
                                        - "v5_ru".
                                        Подробнее о версиях модели Silero TTS см. в файле config.py (modules_t2s).
            silero_language (str): Язык генерации речи для модели Silero TTS (только "ru").
                                   Подробнее см. в файле config.py (modules_t2s).
            silero_speaker (str): Имя встроенного голоса для модели Silero TTS:
                                  - "aidar" - мужской;
                                  - "eugene" — мужской;
                                  - "baya" - женский;
                                  - "kseniya" — женский;
                                  - "xenia" — женский.
            sample_rate (int): Частота дискретизации синтезируемого аудио в Гц для модели Silero TTS:
                               - 8000;
                               - 24000;
                               - 48000.
                               Подробнее о принципах выбора значения см. в файле config.py (modules_t2s).
            put_accent (bool): Автоматическая расстановка ударений для модели Silero TTS:
                               - True - включить;
                               - False - выключить.
            put_yo (bool): Автоматическая замена "е" на "ё", где это нужно, для модели Silero TTS:
                           - True - включить;
                           - False - выключить.
            silero_device (str): Устройство для вычислений, выполняемых моделью Silero TTS:
                                 - "cuda" - GPU;
                                 - "cpu" - CPU.
        """

        # Инициализируем настройки модуля Text2Speech
        self.config = Config(model_name=model_name, qwen_model_name=qwen_model_name,
                             qwen_language=qwen_language, qwen_speaker=qwen_speaker,
                             path_to_audio_for_clone=path_to_audio_for_clone,
                             text_for_clone=text_for_clone, instruct=instruct,
                             qwen_device=qwen_device, dtype=dtype,
                             attn_implementation=attn_implementation,
                             silero_model_version=silero_model_version,
                             silero_language=silero_language, silero_speaker=silero_speaker,
                             sample_rate=sample_rate, put_accent=put_accent, put_yo=put_yo,
                             silero_device=silero_device)

        # Инициализируем модель Text2Speech
        # Если пользователь выбрал модель Qwen3-TTS
        if model_name == "Qwen3-TTS":
            # Инициализируем модель Qwen3-TTS
            self.t2s_model = Qwen3Synthesizer(self.config)
        # Если пользователь выбрал модель Silero TTS
        else:
            # Инициализируем модель Silero TTS
            self.t2s_model = Qwen3Synthesizer(self.config)

        # Инициализируем сборщик синтезированного аудио
        self.builder = AudioBuilder()

    # Функция запуска работы модуля Text2Speech
    def start(self, text: str):
        """
        Args:
            text (str): Текст, который нужно преобразовать в аудио.
        """

        # Синтезируем аудио
        audio, sample_rate = self.t2s_model.synthesis(text)

        # Воспроизводим синтезированное аудио
        AudioPlayer.play_audio(audio, sample_rate)

        # Добавляем синтезированное аудио в сборщик
        self.builder.add_audio(audio, sample_rate)
