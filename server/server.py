import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
    cpu_threads=24,    # 전체의 70~80%
    num_workers=3      # 병렬 chunk 처리
)

executor = ThreadPoolExecutor(max_workers=1)

SAMPLE_RATE = 16000
BUFFER_SECONDS = 1
BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS

async def run_stt(audio):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: model.transcribe(
            audio,
            language="ko",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False
        )
    )

async def handler(ws):
    audio_buffer = np.array([], dtype=np.float32)
    stt_busy = False

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

        # STT 중이면 과거 오디오 버림
        if stt_busy:
            audio_buffer = audio_buffer[-BUFFER_SIZE:]
            continue

        if len(audio_buffer) >= BUFFER_SIZE:
            audio = audio_buffer[:BUFFER_SIZE]
            audio_buffer = audio_buffer[BUFFER_SIZE:]
            stt_busy = True

            async def stt_task(audio):
                nonlocal stt_busy
                try:
                    segments, _ = await run_stt(audio)
                    text = "".join(seg.text for seg in segments).strip()
                    if len(text.replace(" ", "")) >= 3:
                        await ws.send(text)
                finally:
                    stt_busy = False

            asyncio.create_task(stt_task(audio))


async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000, max_size=None):
        print("STT Server started :3000")
        await asyncio.Future()

asyncio.run(main())
