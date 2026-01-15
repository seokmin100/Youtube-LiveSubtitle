import asyncio
import websockets
import json
import numpy as np
from vosk import Model, KaldiRecognizer

MODEL_PATH = "./vosk-model-small-ko-0.22"
SAMPLE_RATE = 16000

model = Model(MODEL_PATH)

async def handler(ws):
    print("Client connected")

    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(False)

    async for message in ws:
        # RTT ping
        if isinstance(message, str):
            if message.startswith("ping:"):
                await ws.send(message)
            continue

        if not isinstance(message, bytes):
            continue

        # PCM Int16 그대로 Vosk에 넣음
        if recognizer.AcceptWaveform(message):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                await ws.send(json.dumps({
                    "type": "final",
                    "text": text
                }))
        else:
            partial = json.loads(recognizer.PartialResult())
            text = partial.get("partial", "").strip()
            if text:
                await ws.send(json.dumps({
                    "type": "partial",
                    "text": text
                }))

async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000, max_size=None):
        print("Vosk STT Server started :3000")
        await asyncio.Future()

asyncio.run(main())
