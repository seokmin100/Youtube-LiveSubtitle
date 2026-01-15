import asyncio
import websockets
import numpy as np
import sqlite3
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# -----------------------------
# Whisper 모델 설정 (CPU 최적화)
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
# SQLite DB 초기화
# -----------------------------
conn = sqlite3.connect("stt_cache.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS subtitles (
    session_id TEXT,
    speaker_id TEXT,
    start_time REAL,
    end_time REAL,
    text TEXT,
    PRIMARY KEY(session_id, start_time)
)
""")
conn.commit()

# -----------------------------
# STT 실행 함수
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
# DB 저장 & 문맥 보정
# -----------------------------
def save_and_correct(session_id, speaker_id, start_time, end_time, text):
    # 기존 문장과 유사하면 merge
    cursor.execute("""
        SELECT text FROM subtitles
        WHERE session_id=? AND speaker_id=? AND end_time>=?
        ORDER BY start_time DESC LIMIT 1
    """, (session_id, speaker_id, start_time))
    row = cursor.fetchone()
    if row:
        prev_text = row[0]
        # 단순 merge: 이전 텍스트 + 현재 텍스트 차이만 추가
        new_text = prev_text + " " + text if text not in prev_text else prev_text
        cursor.execute("""
            UPDATE subtitles SET text=?, end_time=? 
            WHERE session_id=? AND speaker_id=? AND end_time>=?
        """, (new_text, end_time, session_id, speaker_id, start_time))
    else:
        cursor.execute("""
            INSERT OR REPLACE INTO subtitles (session_id, speaker_id, start_time, end_time, text)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, speaker_id, start_time, end_time, text))
    conn.commit()
    return text  # 교정된 텍스트 반환

# -----------------------------
# WebSocket 핸들러
# -----------------------------
async def handler(ws):
    # 클라이언트가 session_id, speaker_id 보내온다고 가정
    initial_msg = await ws.recv()
    if not isinstance(initial_msg, str):
        await ws.close()
        return
    session_id, speaker_id = initial_msg.split(":")
    print(f"Client connected: {session_id}, speaker: {speaker_id}")

    audio_buffer = np.array([], dtype=np.float32)
    start_time = datetime.now().timestamp()

    async for message in ws:
        if isinstance(message, str):
            if message.startswith("ping:"):
                await ws.send(message)
                continue

        if not isinstance(message, bytes):
            continue

        # 오디오 전처리
        chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
        chunk -= np.mean(chunk)
        chunk /= (np.max(np.abs(chunk)) + 1e-6)
        audio_buffer = np.concatenate([audio_buffer, chunk])

        # 슬라이딩 윈도우 처리
        while len(audio_buffer) >= WINDOW_SIZE:
            audio_chunk = audio_buffer[:WINDOW_SIZE]
            audio_buffer = audio_buffer[STEP_SIZE:]

            chunk_start_time = start_time
            start_time += STEP_SECONDS

            async def stt_task(audio_chunk, chunk_start_time):
                try:
                    segments, _ = await run_stt(audio_chunk)
                    text = "".join(seg.text for seg in segments).strip()
                    if len(text.replace(" ", "")) >= 1:
                        # DB 저장 + 문맥/merge 보정
                        corrected = save_and_correct(
                            session_id,
                            speaker_id,
                            chunk_start_time,
                            chunk_start_time + WINDOW_SECONDS,
                            text
                        )
                        await ws.send(corrected)
                except Exception as e:
                    print("STT error:", e)

            asyncio.create_task(stt_task(audio_chunk, chunk_start_time))

# -----------------------------
# 서버 실행
# -----------------------------
async def main():
    async with websockets.serve(handler, "0.0.0.0", 3000, max_size=None):
        print("CPU 32-core 실시간 STT Server started :3000")
        await asyncio.Future()

asyncio.run(main())
