import asyncio
import websockets
import numpy as np
import sqlite3
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

# <CHANGE> CPU 최적화: 모델 크기 선택 (느린 CPU는 "tiny", 빠른 CPU는 "base")
# Tiny: 매우 빠름 (~1초) 하지만 정확도 낮음
# Base: 중간 속도 (~3-5초) 정확도 좋음 - YouTube 수준
model = WhisperModel(
    "base",  # "small" → "base" (CPU에서는 충분한 속도+정확도 균형)
    device="cpu",
    compute_type="int8",  # CPU용 최적화 (int8 양자화)
    cpu_threads=8,  # 32 → 8 (실제 코어 수에 맞춰 조정, 과할당 피함)
    num_workers=1  # 10 → 1 (CPU는 멀티워커 오버헤드 큼)
)

executor = ThreadPoolExecutor(max_workers=1)  # 3 → 1 (CPU 병렬 처리는 역효과)

SAMPLE_RATE = 16000
# <CHANGE> 윈도우 크기 최적화: 더 자주 처리하되 CPU 부하 고려
WINDOW_SECONDS = 1.5  # 3.0 → 1.5 (더 빠른 응답)
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
STEP_SECONDS = 0.3  # 0.8 → 0.3 (더 자주 처리)
STEP_SIZE = int(SAMPLE_RATE * STEP_SECONDS)
MAX_QUEUE_SIZE = 2  # 3 → 2

# SQLite DB
conn = sqlite3.connect("stt_cache.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS subtitles (
    id INTEGER PRIMARY KEY,
    text TEXT UNIQUE,
    count INTEGER DEFAULT 1
)
""")
conn.commit()

# <CHANGE> CPU 최적화 파라미터
async def run_stt(audio, previous_text=""):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: model.transcribe(
            audio,
            language="ko",  # 언어 명시 (성능 향상)
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=250,  # 500 → 250 (반응성 향상)
                threshold=0.5  # VAD 민감도
            ),
            # <CHANGE> CPU를 위해 beam_size 낮춤 (속도 ↑, 정확도는 괜찮음)
            beam_size=5,  # 5 유지 (CPU에서는 이 정도가 최적)
            best_of=3,  # 10 → 3 (CPU 오버헤드 제거)
            temperature=[0.0, 0.1],  # 다양성 유지하되 제한적
            condition_on_previous_text=True,
            initial_prompt=previous_text[-30:] if previous_text else None,  # 문맥 전달
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.3,  # 0.6 → 0.3 (음성 감지 개선)
            # <CHANGE> CPU 최적화 추가 옵션
            repetition_penalty=1.2,  # 반복 감소
            length_penalty=1.0,
            prompt_reset_on_temperature=0.5  # 온도 변화 감지
        )
    )

# <CHANGE> 더 정교한 유사도 기반 교정 (substring 대신 fuzzy matching)
def db_correct(text):
    if not text or len(text.strip()) < 2:
        return text
    
    cursor.execute("SELECT text FROM subtitles ORDER BY count DESC LIMIT 10")
    rows = cursor.fetchall()
    
    best_match = text
    best_ratio = 0
    
    for row in rows:
        cached = row[0]
        # 유사도 계산 (정확한 substring 매칭 대신 fuzzy matching)
        ratio = SequenceMatcher(None, text.lower(), cached.lower()).ratio()
        if ratio > 0.75 and ratio > best_ratio:  # 75% 이상 유사
            best_match = cached
            best_ratio = ratio
    
    # DB에 저장
    cursor.execute(
        "INSERT INTO subtitles (text, count) VALUES (?, 1) ON CONFLICT(text) DO UPDATE SET count = count + 1",
        (text,)
    )
    conn.commit()
    
    return best_match

async def stt_worker(ws, queue):
    previous_text = ""
    while True:
        audio_chunk = await queue.get()
        try:
            segments, _ = await run_stt(audio_chunk, previous_text)
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                corrected = db_correct(text)
                await ws.send(corrected)
                previous_text = text  # 이전 텍스트 저장
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

            # <CHANGE> 오디오 정규화 개선
            chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 정규화: 0으로 나누기 방지
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
        print("CPU Optimized STT Server started on :3000")
        await asyncio.Future()

asyncio.run(main())