import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8",
    cpu_threads=32,
    num_workers=1
)

executor = ThreadPoolExecutor(max_workers=1)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
STEP_SECONDS = 0.2
STEP_SIZE = int(SAMPLE_RATE * STEP_SECONDS)
MAX_QUEUE_SIZE = 1


async def run_stt(audio):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: model.transcribe(
            audio,
            language=None,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=100),
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.8
        )
    )


async def stt_worker(ws, queue):
    while True:
        audio_chunk = await queue.get()
        try:
            segments, _ = await run_stt(audio_chunk)
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                await ws.send(text)
        except websockets.ConnectionClosed:
            print("WebSocket closed")
            break
        finally:
            queue.task_done()

async def clear_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            break

async def handler(ws):
    audio_queue = asyncio.Queue()
    audio_buffer = np.array([], dtype=np.float32)
    worker_task = asyncio.create_task(stt_worker(ws, audio_queue))
    print("Client connected")

    try:
        async for message in ws:
            if isinstance(message, str) and message.startswith("ping:"):
                await ws.send(message)
                continue

            if not isinstance(message, bytes):
                continue

            chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            
            max_val = np.max(np.abs(chunk))
            if max_val > 1e-6:
                chunk = chunk / max_val
            
            audio_buffer = np.concatenate([audio_buffer, chunk])

            while len(audio_buffer) >= WINDOW_SIZE:
                audio_chunk = audio_buffer[:WINDOW_SIZE]
                audio_buffer = audio_buffer[STEP_SIZE:]

                if audio_queue.qsize() >= MAX_QUEUE_SIZE:
                    try:
                        audio_queue.get_nowait()
                        audio_queue.task_done()
                    except asyncio.QueueEmpty:
                        pass

                await audio_queue.put(audio_chunk)

    except websockets.ConnectionClosed:
        print("Client disconnected")
    finally:
        worker_task.cancel()
        await clear_queue(audio_queue)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000, max_size=None):
        print("Speed Optimized STT Server started on :3000")
        await asyncio.Future()

asyncio.run(main())
