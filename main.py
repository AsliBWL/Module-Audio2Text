# pip install faster-whisper sounddevice numpy silero-vad hf-xet

import queue
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

# ========== НАСТРОЙКИ ==========
SAMPLE_RATE = 16000  # Частота дискретизации (16 кГц)
VAD_FRAME_SAMPLES = 512  # Размер фрейма для VAD (32 мс при 16 кГц)
BYTES_PER_SAMPLE = 2  # int16 = 2 байта
FRAME_BYTES_LEN = VAD_FRAME_SAMPLES * BYTES_PER_SAMPLE

WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large-v2
DEVICE = "cpu"  # "cuda" для GPU, "cpu" для CPU
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"  # float16 для GPU, int8 для CPU
# ===============================

# Очередь для аудиоданных от микрофона
audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """Callback-функция, вызываемая при поступлении аудио с микрофона"""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))


def pcm16_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Конвертирует PCM int16 в float32 (диапазон -1..1)"""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


# Инициализация модели faster-whisper
print(f"Загрузка модели faster-whisper ({WHISPER_MODEL_SIZE}) на {DEVICE}...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

# Инициализация Silero VAD
vad_model = load_silero_vad(onnx=True)
vad_iterator = VADIterator(
    vad_model,
    threshold=0.5,  # Порог срабатывания (0-1)
    sampling_rate=SAMPLE_RATE,
    min_silence_duration_ms=300,  # Минимальная тишина для завершения фразы
    speech_pad_ms=400,  # Добавляем паузу вокруг речи
)

# Запуск захвата с микрофона
with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=VAD_FRAME_SAMPLES,
        device=None,  # Использовать устройство по умолчанию
        dtype="int16",
        channels=1,
        callback=audio_callback,
):
    print("🎤 Слушаю... Говорите в микрофон. Ctrl+C для выхода.")
    print("-" * 50)

    pending_float = np.array([], dtype=np.float32)
    pending_bytes = bytearray()
    in_speech = False
    speech_segment = bytearray()

    try:
        while True:
            # Получаем аудио из очереди
            data = audio_queue.get()

            # Добавляем данные в буферы
            pending_float = np.concatenate([pending_float, pcm16_to_float32(data)])
            pending_bytes.extend(data)

            # Обрабатываем пока есть полный фрейм
            while len(pending_float) >= VAD_FRAME_SAMPLES and len(pending_bytes) >= FRAME_BYTES_LEN:
                frame_float = pending_float[:VAD_FRAME_SAMPLES]
                pending_float = pending_float[VAD_FRAME_SAMPLES:]

                frame_bytes = bytes(pending_bytes[:FRAME_BYTES_LEN])
                del pending_bytes[:FRAME_BYTES_LEN]

                event = vad_iterator(frame_float)

                # Начало речи
                if event is not None and "start" in event and not in_speech:
                    in_speech = True
                    speech_segment = bytearray()
                    print("\n🎙️ Речь началась...")

                # Накопление речи
                if in_speech:
                    speech_segment.extend(frame_bytes)

                # Конец речи
                if event is not None and "end" in event and in_speech:
                    print("🛑 Речь закончилась. Распознаю...")

                    if len(speech_segment) > 0:
                        # Конвертируем накопленные данные в numpy массив для whisper
                        audio_np = np.frombuffer(bytes(speech_segment), dtype=np.int16).astype(np.float32) / 32768.0

                        # Распознавание через faster-whisper
                        segments, info = whisper_model.transcribe(
                            audio_np,
                            beam_size=5,
                            language=None,  # Автоопределение языка
                            vad_filter=False,  # VAD уже используем отдельно
                        )

                        # Собираем результат
                        transcribed_text = " ".join(segment.text for segment in segments)
                        print(f"📝 [{info.language}:{info.language_probability:.2f}] {transcribed_text}")
                        print("-" * 50)

                    # Сброс состояния VAD
                    vad_iterator.reset_states()
                    in_speech = False
                    speech_segment = bytearray()

    except KeyboardInterrupt:
        print("\n👋 Программа остановлена.")