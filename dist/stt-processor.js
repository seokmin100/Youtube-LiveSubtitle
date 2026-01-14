class STTProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    let sum = 0;
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * input[i];
    }
    const rms = Math.sqrt(sum / input.length);

    // 🔥 너무 작은 소리는 버림
    if (rms < 0.01) return true;

    const buffer = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      buffer[i] = Math.max(-1, Math.min(1, input[i])) * 32767;
    }

    this.port.postMessage({
      audio: buffer.buffer,
      rms
    }, [buffer.buffer]);

    return true;
  }
}

registerProcessor("stt-processor", STTProcessor);
