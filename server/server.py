import asyncio
import websockets
import numpy as np
import sqlite3
import re
import time
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Whisper 모델 설정
# -----------------------------
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
    cpu_threads=32,
    num_workers=10
)

executor = ThreadPoolExecutor(max_workers=3)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.2
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
STEP_SECONDS = 0.5
STEP_SIZE = int(SAMPLE_RATE * STEP_SECONDS)
MAX_QUEUE_SIZE = 4

# -----------------------------
# SQLite DB 초기화
# -----------------------------
conn = sqlite3.connect("stt_cache.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS subtitles (
    text TEXT PRIMARY KEY,
    confidence INTEGER DEFAULT 1
)
""")
conn.commit()

# -----------------------------
# 유틸
# -----------------------------
last_commit_time = {}
last_sent_text = None

def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\s가-힣]", "", text)
    return text

# -----------------------------
# STT 실행
# -----------------------------
async def run_stt(audio):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: model.transcribe(
            audio,
            language="ko",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=600),
            beam_size=3,
            best_of=2,
            temperature=0.0,
            condition_on_previous_text=True
        )
    )

# -----------------------------
# DB 로직
# -----------------------------
def db_commit(text, interval=1.0):
    now = time.time()
    if now - last_commit_time.get(text, 0) < interval:
        return
    last_commit_time[text] = now

    cursor.execute("""
        INSERT INTO subtitles (text, confidence)
        VALUES (?, 1)
        ON CONFLICT(text)
        DO UPDATE SET confidence = confidence + 1
    """, (text,))
    conn.commit()

def db_is_stable(text, threshold=3):
    cursor.execute(
        "SELECT confidence FROM subtitles WHERE text = ?",
        (text,)
    )
    row = cursor.fetchone()
    return row is not None and row[0] >= threshold

# -----------------------------
# STT Worker
# -----------------------------
async def stt_worker(ws, queue):
    global last_sent_text

    while True:
        audio_chunk = await queue.get()
        try:
            segments, _ = await run_stt(audio_chunk)

            for seg in segments:
                raw = seg.text
                if not raw or seg.no_speech_prob >= 0.3:
                    continue

                text = normalize(raw)
                if not text or len(text) < 2:
                    continue

                db_commit(text)
                stable = db_is_stable(text)

                out = text if stable else f"~{text}"

                if out != last_sent_text:
                    await ws.send(out)
                    last_sent_text = out

        except websockets.ConnectionClosed:
            break
        finally:
            queue.task_done()

# -----------------------------
# 큐 정리
# -----------------------------
async def clear_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            break

# -----------------------------
# WebSocket 핸들러
# -----------------------------
async def handler(ws):
    audio_queue = asyncio.Queue()
    audio_buffer = np.array([], dtype=np.float32)
    worker_task = asyncio.create_task(stt_worker(ws, audio_queue))

    try:
        async for message in ws:
            if isinstance(message, str) and message.startswith("ping:"):
                await ws.send(message)
                continue

            if not isinstance(message, bytes):
                continue

            chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            chunk -= np.mean(chunk)
            chunk /= (np.max(np.abs(chunk)) + 1e-6)

            audio_buffer = np.concatenate([audio_buffer, chunk])

            while len(audio_buffer) >= WINDOW_SIZE:
                audio_chunk = audio_buffer[:WINDOW_SIZE]
                audio_buffer = audio_buffer[STEP_SIZE:]

                if audio_queue.qsize() >= MAX_QUEUE_SIZE:
                    await audio_queue.get()
                    audio_queue.task_done()

                await audio_queue.put(audio_chunk)

    except websockets.ConnectionClosed:
        pass
    finally:
        worker_task.cancel()
        await clear_queue(audio_queue)

# -----------------------------
# 서버 실행
# -----------------------------
async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000, max_size=None):
        print("STT Server started :3000")
        await asyncio.Future()

asyncio.run(main())
