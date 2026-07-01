import asyncio
import websockets
import soundfile as sf
import numpy as np


async def test():
    # Генерируем тестовый сигнал (1 секунда тишины с коротким пиком)
    audio = np.zeros(16000, dtype=np.float32)
    audio[8000:8100] = 0.5

    # Конвертируем в WAV
    import io
    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, format='WAV')
    wav_bytes = buffer.getvalue()

    # Отправляем на сервер
    async with websockets.connect("ws://192.168.1.100:9090") as ws:
        await ws.send(wav_bytes)
        response = await ws.recv()
        print("Ответ сервера:", response)


asyncio.run(test())
