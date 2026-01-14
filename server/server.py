import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BUFFER_SECONDS = 1.5   # 🔥 작을수록 실시간
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_SECONDS)

# CPU 최적화 모델
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"   # ⭐ 핵심 (속도 대폭 ↑)
)

audio_buffer = np.empty(0, dtype=np.float32)

async def handler(ws):
    global audio_buffer
    print("Client connected")

    async for message in ws:
        if not isinstance(message, bytes):
            continue

        chunk = np.frombuffer(message, dtype=np.float32)
        audio_buffer = np.concatenate([audio_buffer, chunk])

        if len(audio_buffer) < BUFFER_SIZE:
            continue

        audio = audio_buffer[:BUFFER_SIZE]
        audio_buffer = audio_buffer[BUFFER_SIZE:]

        # 🔥 faster-whisper는 바로 numpy 입력 가능
        segments, info = model.transcribe(
            audio,
            language=None,        # 자동 언어 감지
            vad_filter=True,      # 무음 제거
            beam_size=1           # 실시간용
        )

        text = ""
        for seg in segments:
            text += seg.text

        text = text.strip()
        if text:
            await ws.send(text)

async def main():
    async with websockets.serve(
        handler,
        "0.0.0.0",
        3000,
        max_size=None
    ):
        print("🚀 Faster-Whisper STT Server started :3000")
        await asyncio.Future()

asyncio.run(main())
