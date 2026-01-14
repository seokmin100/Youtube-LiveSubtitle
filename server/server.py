import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel
import time

# 모델 로드 (처음 1번만)
model = WhisperModel(
    "small",          # tiny / base / small / medium
    device="cpu",     # gpu 있으면 "cuda"
    compute_type="int8"
)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 3  # 3초 단위 STT

async def handler(ws):
    print("클라이언트 연결")
    audio_buffer = []

    async for message in ws:
        # PCM 16bit → float32 변환
        pcm = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
        audio_buffer.append(pcm)

        length = sum(len(c) for c in audio_buffer)
        if length >= SAMPLE_RATE * CHUNK_SECONDS:
            audio = np.concatenate(audio_buffer)
            audio_buffer.clear()

            segments, _ = model.transcribe(
                audio,
                language="ko",
                task="translate",   # 번역 원하면 "translate"
                vad_filter=True
            )

            for seg in segments:
                text = seg.text.strip()
                if text:
                    await ws.send(text)
                    print(">>", text)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000):
        print("Whisper STT 서버 실행중")
        await asyncio.Future()

asyncio.run(main())