import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"  # CPU 필수
)

SAMPLE_RATE = 16000
BUFFER_SECONDS = 2
BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS

audio_buffer = np.array([], dtype=np.float32)

async def handler(ws):
    global audio_buffer
    print("Client connected")

    async for message in ws:
        if not isinstance(message, bytes):
            continue

        # ⭐ Int16 PCM → float32
        chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
        audio_buffer = np.concatenate([audio_buffer, chunk])

        if len(audio_buffer) >= BUFFER_SIZE:
            audio = audio_buffer[:BUFFER_SIZE]
            audio_buffer = audio_buffer[BUFFER_SIZE:]

            segments, _ = model.transcribe(
                audio,
                language="ko",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False
            )

            text = "".join(seg.text for seg in segments).strip()
            if text:
                print("STT:", text)
                await ws.send(text)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000, max_size=None):
        print("STT Server started :3000")
        await asyncio.Future()

asyncio.run(main())
