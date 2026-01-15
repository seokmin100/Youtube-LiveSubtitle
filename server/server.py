import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
    cpu_threads=24,    # 전체의 70~80%
    num_workers=3      # 병렬 chunk 처리
)

SAMPLE_RATE = 16000
BUFFER_SECONDS = 1
BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS

async def handler(ws):
    audio_buffer = np.array([], dtype=np.float32)
    print("Client connected")

    async for message in ws:
        if isinstance(message, str):
            if message.startswith("ping:"):
                await ws.send(message)
        
        if not isinstance(message, bytes):
            continue

        # Int16 PCM → float32
        chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
        audio_buffer = np.concatenate([audio_buffer, chunk])

        if len(audio_buffer) >= BUFFER_SIZE:
            audio = audio_buffer[:BUFFER_SIZE]
            audio_buffer = audio_buffer[BUFFER_SIZE:]

            segments, _ = model.transcribe(
                audio,
                language="ko",
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300
                ),
                condition_on_previous_text=False,
                beam_size=1,
                # best_of=5,
                temperature=0.0
            )

            text = "".join(seg.text for seg in segments).strip()

            if len(text.replace(" ", "")) < 3:
                continue

            await ws.send(text)


async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000, max_size=None):
        print("STT Server started :3000")
        await asyncio.Future()

asyncio.run(main())
