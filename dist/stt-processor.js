// ---------------- STT AudioWorklet Processor ----------------
class STTProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.pcmBuffer = [];
    this.pcmLength = 0;
    this.TARGET_SAMPLES = 16000 * 0.25; // 0.25초마다 서버로 전송
    this.RMS_THRESHOLD = 0.01;        // 너무 작은 소리 무시
    this.silenceFrames = 0;         // 무음 프레임 카운터
    this.MAX_SILENCE_FRAMES = 10;   // 연속 무음 허용 횟수 (~0.25초)
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    // RMS 계산
    let sum = 0;
    for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
    const rms = Math.sqrt(sum / input.length);

    // 무음 감지 - 바로 버리지 않고 카운터 사용
    if (rms < this.RMS_THRESHOLD) {
      this.silenceFrames++;
      // 연속 무음이 너무 길면 버퍼 비우고 스킵
      if (this.silenceFrames > this.MAX_SILENCE_FRAMES) {
        if (this.pcmLength > 0) {
          // 남은 버퍼 전송 후 초기화
          this._flushBuffer(rms);
        }
        return true;
      }
      // 짧은 무음은 계속 버퍼링 (말 중간 쉼 보존)
    } else {
      this.silenceFrames = 0;
    }

    // Float32 → Int16 변환
    const clamped = Float32Array.from(input, x => Math.max(-1, Math.min(1, x)));
    const buffer = new Int16Array(clamped.length);
    for (let i = 0; i < clamped.length; i++) buffer[i] = clamped[i] * 32767;

    // 버퍼 누적
    this.pcmBuffer.push(buffer);
    this.pcmLength += buffer.length;

    // 일정 샘플 이상 모이면 서버로 전송
    if (this.pcmLength >= this.TARGET_SAMPLES) {
      const merged = new Int16Array(this.pcmLength);
      let offset = 0;
      for (const buf of this.pcmBuffer) {
        merged.set(buf, offset);
        offset += buf.length;
      }

      // 서버 전송
      this.port.postMessage({
        audio: merged.buffer,
        rms
      }, [merged.buffer]);

      // 버퍼 초기화
      this.pcmBuffer = [];
      this.pcmLength = 0;
    }

    return true;
  }

  _flushBuffer(rms) {
    if (this.pcmLength === 0) return;
    
    const merged = new Int16Array(this.pcmLength);
    let offset = 0;
    for (const buf of this.pcmBuffer) {
      merged.set(buf, offset);
      offset += buf.length;
    }
    this.port.postMessage({ audio: merged.buffer, rms }, [merged.buffer]);
    this.pcmBuffer = [];
    this.pcmLength = 0;
  }
}

registerProcessor("stt-processor", STTProcessor);
