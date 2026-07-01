# Импорт библиотек
from modules_s2t.config import Config
from modules_s2t.audio_capture import MicrophoneSource
from modules_s2t.vad_detector import SileroVADDetector
from modules_s2t.speech_recognizer import FasterWhisperRecognizer, RemoteWhisperRecognizer
from modules_s2t.speech_builder import SpeechBuilder
from modules_s2t.session_context import SessionContext


class Context:
    """
    Класс Context объединяет в себе все объекты, необходимые для захвата и обработки аудио:
    - объект класса Config: хранение значений параметров настройки приложения;
    - объект класса AudioSource: захват аудио;
    - объект класса AudioConverter: конвертация аудиоданных;
    - объект класса SessionContext: поддержка состояния сессии распознавания;
    - объект класса VADInterface: детекция голосовой активности;
    - объект класса SpeechRecognizerInterface: преобразование аудио в текст;
    - объект класса SpeechBuilder: сборка итогового текста.
    """

    # Конструктор
    def __init__(self, whisper_model_version: str = "small",
                 device: str = "cpu", vad_threshold: float = 0.6,
                 min_silence_duration_ms: int = 500, beam_size: int = 5,
                 auto_language: bool = False, language: str = "ru",
                 compute_type: str = None):
        """
        Args:
            whisper_model_version (str): Версия модели Whisper ("tiny", "base", "small", "medium", "large", "turbo").
                                         "small" - на ней WER и CER существенно уменьшаются
                                         (т. е. улучшается качество распознавания).
            device (str): Устройство для вычислений ("cpu", "cuda", "auto" (сам определяет лучшее устройство)).
                          "cpu", т. к. пока код запускается на устройстве без GPU, а вообще "cuda".
            vad_threshold (float): Порог срабатывания модели детектора речи.
                                   Диапазон: 0.0 (очень чувствительный) - 1.0 (очень нечувствительный).
                                   Правила выбора:
                                   - 0.5 - стандартное значение (хороший баланс);
                                   - Меньше 0.5 - будет реагировать на тихие звуки, шёпот, может ловить шумы как речь;
                                   - Больше 0.5 - будет игнорировать тихую речь, но зато меньше ложных срабатываний;
                                   - Для шумного цеха рекомендую 0.6-0.7, для тихого кабинета 0.4-0.5.
            min_silence_duration_ms (int): Минимальная длительность тишины для завершения фразы, которая влияет на то,
                                           как быстро запускается распознавание после паузы (в мс).
                                           Правила выбора:
                                           - Меньше 300 мс - быстрее реакция, но может разрезать длинные слова с паузой;
                                           - Больше 300 мс - реже ошибается, но дольше ждать после окончания речи;
                                           - Для быстрого диалога: 200-300 мс, для спокойной речи: 500-700 мс.
            beam_size (int): Количество наиболее вероятных вариантов расшифровки аудио, рассматриваемых параллельно.
                             Чем больше число, тем выше качество, но медленнее скорость распознавания
                             Правила выбора:
                             - 5 - хороший баланс производительности и качества;
                             - 1 - самый быстрый, но может ошибаться на сложных фразах;
                             - 10-15 - максимальное качество, но медленнее в 2-3 раза;
                             - Для реального времени: 3-5, для пакетной обработки: 10-15.
            auto_language (bool): Автоопределение языка речи:
                                  - True - модель сама определяет язык (русский, английский и т. д.);
                                  - False: нужно явно указать язык (например, language="ru").
                                  Правила выбора:
                                  - True - удобно, когда язык заранее неизвестен; может переключаться между
                                  языками в одном аудиопотоке;
                                  - False - быстрее (не тратит время на определение языка);
                                  точнее для многоязычных моделей.
            language (str): Язык, на котором будет распознаваться речь (если auto_language = False):
                            - "ru" - русский;
                            - "en" - английский;
                            - "de" - немецкий;
                            - "fr" - французский;
                            - "es" - испанский;
                            - "it" - итальянский;
                            - "pt" - португальский;
                            - "zh" - китайский;
                            - "ja" - японский;
                            - "ko" - корейский;
                            - "ar" - арабский;
                            - "hi" - хинди;
                            - "tr" - турецкий;
                            - "pl" - польский;
                            - "uk" - украинский;
                            - "kk" - казахский;
                            - "be" - белорусский;
                            - "sv" - шведский;
                            - "no" - норвежский;
                            - "da" - датский;
                            - "nl" - нидерландский;
                            - "fi" - финский;
                            - "cs" - чешский;
                            - "hu" - венгерский;
                            - "ro" - румынский;
                            - "bg" - болгарский;
                            - "el" - греческий;
                            - "he" - иврит;
                            - "th" - тайский;
                            - "vi" - вьетнамский.
            compute_type (str): Тип вычислений, от которого зависит точность модели:
                                - "float32" (32-битные числа с плавающей точкой) - максимальная точность,
                                требует больше памяти;
                                - "float16" (16-битные числа с плавающей точкой) - хороший баланс для GPU
                                (рекомендуется для "cuda");
                                - "int8" (8-битные целые числа) - ускорение в 2-4 раза на CPU, небольшая
                                потеря качества;
                                - None - тип будет выбран автоматически.
                                Правила выбора:
                                - На CPU лучше всего работает "int8" (самое быстрое, память ~в 4 раза меньше float32);
                                - На GPU лучше всего "float16" (быстрее float32 в 2 раза, память в 2 раза меньше);
                                - "float32" нужен только если модель дает ошибки при меньшей точности;
                                - None: автоматически выберет для GPU → "float16", для CPU → "int8".
        """
        
        # Инициализируем настройки приложения
        self.config = Config(whisper_model_version=whisper_model_version,
                             device=device, vad_threshold=vad_threshold,
                             min_silence_duration_ms=min_silence_duration_ms,
                             beam_size=beam_size, auto_language=auto_language,
                             language=language, compute_type=compute_type)

        # Создаем объект для поддержания состояния сессии распознавания
        self.session = SessionContext(self.config)

        # Создаем детектор голосовой активности
        self.vad = SileroVADDetector(self.config)

        # Создаем распознаватель речи (преобразует аудио в текст)
        self.recognizer = FasterWhisperRecognizer(self.config)
        # self.recognizer = RemoteWhisperRecognizer(self.config)

        # Создаем сборщик итогового текста
        self.builder = SpeechBuilder()

        # Создаем объект для захвата аудио с микрофона (без callback, всё работает через очередь)
        """
        Данные складываются в очередь, потому что, если callback долгий - можно потерять аудио.
        """
        self.mic = MicrophoneSource(self.config)

    # Функция запуска работы приложения
    def start(self):
        """
        Остановить работу приложения можно комбинацией клавиш Ctrl+C.
        """

        # Пытаемся
        try:
            # Запустить захват аудио с микрофона
            self.mic.start()
            # print("🎤 Слушаю... Говорите в микрофон. Ctrl+C для выхода.")
            # print("-" * 50)

            # Пока захват аудио с микрофона нахожится в запущенном состоянии
            while self.mic.is_running():
                # Получаем следующий фрейм с аудиоданными из очереди
                chunk = self.mic.get_audio_chunk()

                # Если очередь пустая
                if chunk is None:
                    # Завершаем текущую итерацию цикла
                    continue

                # Добавляем сырой байтовый фрейм от микрофона в буфер
                self.session.add_audio_chunk(chunk)

                # Пока в буфере достаточно данных, чтобы извлечь из него хотя бы 1 фрейм
                while self.session.has_complete_frame():
                    # Извлекаем следующий фрейм из буфера
                    frame_float, frame_bytes = self.session.pop_frame()

                    # Подаем извлеченный фрейм на VAD для обработки
                    event = self.vad.process_frame(frame_float)

                    # Если:
                    # 1. VAD обнаружил изменения (появилась или исчезла речь)
                    # 2. изменение - это начало речи
                    # 3. сессия находится не в режиме записи речи
                    if event is not None and "start" in event and not self.session.in_speech:
                        # Начинаем запись речи
                        self.session.start_speech()
                        # print("\n🎙️ Речь началась...")

                    # Если сессия находится в режиме записи речи
                    if self.session.in_speech:
                        # Добавляем фрейм в накопитель фразы
                        self.session.accumulate_speech(frame_bytes)

                    # Если:
                    # 1. VAD обнаружил изменения (появилась или исчезла речь)
                    # 2. изменение - это конец речи
                    # 3. сессия находится в режиме записи речи
                    if event is not None and "end" in event and self.session.in_speech:
                        # print("🛑 Речь закончилась. Распознаю...")

                        # Получаем накопленную фразу
                        audio_for_recognition = self.session.end_speech()

                        # Если длина массива с аудиоданными не нулевая
                        if len(audio_for_recognition) > 0:
                            # Преобразуем аудио в текст и получаем текст, язык и уверенность в языке
                            text, lang, lang_prob = self.recognizer.recognize(audio_for_recognition)
                            # Добавляем распознанный текст в сборщик итогового текста
                            self.builder.add_fragment(text)
                            # print(f"📝 [{lang}:{lang_prob:.2f}] {text}")
                            # print("-" * 50)

                        # Сбрасываем внутреннее состояние VAD
                        self.vad.reset()
        # Перехватываем исключение, которое возникает при нажатии Ctrl+C
        except KeyboardInterrupt:
            # print("\n👋 Программа остановлена.")
            pass
        # В любом случае
        finally:
            # Останавливаем захват аудио с микрофона
            self.mic.stop()
