import asyncio
import websockets
import numpy as np
import soundfile as sf
import whisper
import os

model = whisper.load_model("base")

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

        # Float32 PCM
        chunk = np.frombuffer(message, dtype=np.float32)
        audio_buffer = np.concatenate([audio_buffer, chunk])

        if len(audio_buffer) >= BUFFER_SIZE:
            audio = audio_buffer[:BUFFER_SIZE]
            audio_buffer = audio_buffer[BUFFER_SIZE:]

            os.makedirs("temp", exist_ok=True)
            wav_path = "temp/audio.wav"

            # WAV 저장
            sf.write(wav_path, audio, SAMPLE_RATE)

            # Whisper STT
            result = model.transcribe(wav_path, language="en")
            text = result.get("text", "").strip()

            if text:
                await ws.send(text)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000, max_size=None):
        print("STT Server started :3000")
        await asyncio.Future()

asyncio.run(main())
