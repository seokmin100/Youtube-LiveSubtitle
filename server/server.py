import asyncio
import websockets
import numpy as np
import sqlite3
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# -----------------------------
# Whisper 모델 설정
# -----------------------------
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
    cpu_threads=32,
    num_workers=3
)

executor = ThreadPoolExecutor(max_workers=3)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.2
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
STEP_SECONDS = 0.5
STEP_SIZE = int(SAMPLE_RATE * STEP_SECONDS)

# -----------------------------
# SQLite DB 초기화 (모든 영상/화자 공유)
# -----------------------------
conn = sqlite3.connect("stt_cache.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS subtitles (
    text TEXT PRIMARY KEY
)
""")
conn.commit()

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
# DB 기반 자동 교정
# -----------------------------
def db_correct(text):
    # 비슷한 텍스트가 DB에 있으면 교정
    cursor.execute("SELECT text FROM subtitles")
    rows = cursor.fetchall()
    for row in rows:
        cached = row[0]
        # 단순 유사도 비교 (여기선 포함 여부)
        if text in cached or cached in text:
            return cached
    # DB에 없는 경우 새로 추가
    cursor.execute("INSERT OR IGNORE INTO subtitles (text) VALUES (?)", (text,))
    conn.commit()
    return text

# -----------------------------
# WebSocket 핸들러
# -----------------------------
async def handler(ws):
    audio_buffer = np.array([], dtype=np.float32)
    stt_busy = False
    print("Client connected")

    async for message in ws:
        if isinstance(message, str):
            if message.startswith("ping:"):
                await ws.send(message)
                continue

        if not isinstance(message, bytes):
            continue

        # Int16 PCM → float32 + normalization
        chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
        chunk -= np.mean(chunk)
        chunk /= (np.max(np.abs(chunk)) + 1e-6)
        audio_buffer = np.concatenate([audio_buffer, chunk])

        # STT 중이면 과거 오디오 버림
        if stt_busy:
            audio_buffer = audio_buffer[-WINDOW_SIZE:]
            continue

        # 슬라이딩 윈도우 처리
        while len(audio_buffer) >= WINDOW_SIZE:
            audio_chunk = audio_buffer[:WINDOW_SIZE]
            audio_buffer = audio_buffer[STEP_SIZE:]
            stt_busy = True

            async def stt_task(audio_chunk):
                nonlocal stt_busy
                try:
                    segments, _ = await run_stt(audio_chunk)
                    text = "".join(seg.text for seg in segments).strip()
                    if len(text.replace(" ", "")) >= 1:
                        corrected = db_correct(text)
                        await ws.send(corrected)
                finally:
                    stt_busy = False

            asyncio.create_task(stt_task(audio_chunk))

# -----------------------------
# 서버 실행
# -----------------------------
async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000, max_size=None):
        print("CPU 32-core STT Server + DB Auto-Correct started :3000")
        await asyncio.Future()

asyncio.run(main())
