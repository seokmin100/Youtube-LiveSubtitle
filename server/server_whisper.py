import asyncio
import websockets
import json
import numpy as np
from faster_whisper import WhisperModel

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
CHUNK_SEC = 1.0
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_SEC)

# ---------------- MODEL ----------------
whisper = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
    cpu_threads=24,
    num_workers=3
)

# ---------------- WORKER ----------------
async def whisper_worker(queue: asyncio.Queue, ws):
    while True:
        audio = await queue.get()
        try:
            segments, _ = whisper.transcribe(
                audio,
                language="ko",
                beam_size=1,
                best_of=3,
                temperature=0.0,
                condition_on_previous_text=False
            )
            text = "".join(s.text for s in segments).strip()
            if text:
                await ws.send(json.dumps({
                    "type": "final",
                    "text": text
                }))
        except Exception as e:
            print("Whisper error:", e)

# ---------------- WS ----------------
async def handler(ws):
    print("Client connected")

    buffer = np.zeros(0, dtype=np.float32)
    queue = asyncio.Queue(maxsize=1)

    asyncio.create_task(whisper_worker(queue, ws))

    async for message in ws:
        if not isinstance(message, bytes):
            continue

        chunk = np.frombuffer(message, np.int16).astype(np.float32) / 32768.0
        buffer = np.concatenate([buffer, chunk])

        if len(buffer) >= CHUNK_SIZE:
            audio = buffer[-CHUNK_SIZE:]  # 최신만
            buffer = buffer[-CHUNK_SIZE:] # ring

            if queue.full():
                _ = queue.get_nowait()

            await queue.put(audio)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000, max_size=None):
        print("Whisper-only STT server :3000")
        await asyncio.Future()

asyncio.run(main())
