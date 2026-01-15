import asyncio
import websockets
import json
import numpy as np
from vosk import Model as VoskModel, KaldiRecognizer
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
WHISPER_BUFFER_SEC = 1.2
WHISPER_BUFFER_SIZE = int(SAMPLE_RATE * WHISPER_BUFFER_SEC)

# ---------------- MODELS ----------------
vosk_model = VoskModel("./vosk-model-small-ko-0.22")

whisper_model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
    cpu_threads=24,
    num_workers=3
)

executor = ThreadPoolExecutor(max_workers=1)

# ---------------- ASYNC WHISPER ----------------
async def run_whisper(audio: np.ndarray):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: whisper_model.transcribe(
            audio,
            language="ko",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=True
        )
    )

# ---------------- WS HANDLER ----------------
async def handler(ws):
    print("Client connected")

    vosk = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    vosk.SetWords(False)

    whisper_buffer = np.array([], dtype=np.float32)

    async for message in ws:
        # RTT ping
        if isinstance(message, str):
            if message.startswith("ping:"):
                await ws.send(message)
            continue

        if not isinstance(message, bytes):
            continue

        # ---------------- VOSK (IMMEDIATE) ----------------
        if vosk.AcceptWaveform(message):
            res = json.loads(vosk.Result())
            if res.get("text"):
                await ws.send(json.dumps({
                    "type": "final_vosk",
                    "text": res["text"]
                }))
        else:
            partial = json.loads(vosk.PartialResult()).get("partial", "")
            if partial:
                await ws.send(json.dumps({
                    "type": "partial",
                    "text": partial
                }))

        # ---------------- WHISPER BUFFER ----------------
        chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
        whisper_buffer = np.concatenate([whisper_buffer, chunk])

        if len(whisper_buffer) >= WHISPER_BUFFER_SIZE:
            audio = whisper_buffer[:WHISPER_BUFFER_SIZE]
            whisper_buffer = whisper_buffer[WHISPER_BUFFER_SIZE:]

            segments, _ = await run_whisper(audio)
            text = "".join(seg.text for seg in segments).strip()

            if len(text.replace(" ", "")) >= 3:
                await ws.send(json.dumps({
                    "type": "final",
                    "text": text
                }))

async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000, max_size=None):
        print("Hybrid STT Server started :3000")
        await asyncio.Future()

asyncio.run(main())
