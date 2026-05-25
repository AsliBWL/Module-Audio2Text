# Импорт библиотек
from config import Config
from audio_capture import MicrophoneSource
from vad_detector import SileroVADDetector
from speech_recognizer import FasterWhisperRecognizer
from speech_builder import SpeechBuilder
from session_context import SessionContext


# Основная функция
def main():
    # Инициализируем настройки приложения
    config = Config()

    # Создаем объект для поддержания состояния сессии распознавания
    session = SessionContext(config)

    # Создаем детектор речи (VAD)
    vad = SileroVADDetector(config)

    # Создаем распознаватель речи (преобразует аудио в текст)
    recognizer = FasterWhisperRecognizer(config)

    # Создаем сборщик итогового текста
    builder = SpeechBuilder()

    # Создаем объект для захвата аудио с микрофона (без callback, всё работает через очередь)
    """
    Данные складываются в очередь, потому что, если callback долгий - можно потерять аудио.
    """
    mic = MicrophoneSource(config)

    # Пытаемся
    try:
        # Запустить захват аудио с микрофона
        mic.start()
        print("🎤 Слушаю... Говорите в микрофон. Ctrl+C для выхода.")
        print("-" * 50)

        # Пока захват аудио с микрофона нахожится в запущенном состоянии
        while mic.is_running():
            # Получаем следующий фрейм с аудиоданными из очереди
            chunk = mic.get_audio_chunk()

            # Если очередь пустая
            if chunk is None:
                # Завершаем текущую итерацию цикла
                continue

            # Добавляем сырой байтовый фрейм от микрофона в буфер
            session.add_audio_chunk(chunk)

            # Пока в буфере достаточно данных, чтобы извлечь из него хотя бы 1 фрейм
            while session.has_complete_frame():
                # Извлекаем следующий фрейм из буфера
                frame_float, frame_bytes = session.pop_frame()

                # Подаем извлеченный фрейм на VAD для обработки
                event = vad.process_frame(frame_float)

                # Если:
                # 1. VAD обнаружил изменения (появилась или исчезла речь)
                # 2. изменение - это начало речи
                # 3. сессия находится не в режиме записи речи
                if event is not None and "start" in event and not session.in_speech:
                    # Начинаем запись речи
                    session.start_speech()
                    print("\n🎙️ Речь началась...")

                # Если сессия находится в режиме записи речи
                if session.in_speech:
                    # Добавляем фрейм в накопитель фразы
                    session.accumulate_speech(frame_bytes)

                # Если:
                # 1. VAD обнаружил изменения (появилась или исчезла речь)
                # 2. изменение - это конец речи
                # 3. сессия находится в режиме записи речи
                if event is not None and "end" in event and session.in_speech:
                    print("🛑 Речь закончилась. Распознаю...")

                    # Получаем накопленную фразу
                    audio_for_recognition = session.end_speech()

                    # Если длина массива с аудиоданными не нулевая
                    if len(audio_for_recognition) > 0:
                        # Преобразуем аудио в текст и получаем текст, язык и уверенность в языке
                        text, lang, lang_prob = recognizer.recognize(audio_for_recognition)
                        # Добавляем распознанный текст в сборщик итогового текста
                        builder.add_fragment(text)
                        print(f"📝 [{lang}:{lang_prob:.2f}] {text}")
                        print("-" * 50)

                    # Сбрасываем внутреннее состояние VAD
                    vad.reset()
    # Перехватываем исключение, которое возникает при нажатии Ctrl+C
    except KeyboardInterrupt:
        print("\n👋 Программа остановлена.")
    # В любом случае
    finally:
        # Останавливаем захват аудио с микрофона
        mic.stop()


# Если этот файл был запущен как самостоятельная программа
if __name__ == "__main__":
    # Выполняем функцию main
    main()
