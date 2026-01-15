// ---------------- STT AudioWorklet Processor ----------------
class STTProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.pcmBuffer = [];
    this.pcmLength = 0;
    this.TARGET_SAMPLES = 16000 * 0.5; // 0.5초마다 서버로 전송
    this.RMS_THRESHOLD = 0.005;        // 너무 작은 소리 무시
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    // RMS 계산
    let sum = 0;
    for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
    const rms = Math.sqrt(sum / input.length);

    // 너무 작은 소리는 전송하지 않음
    if (rms < this.RMS_THRESHOLD) return true;

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
}

registerProcessor("stt-processor", STTProcessor);
