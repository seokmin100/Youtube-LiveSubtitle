class STTProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0][0];
    if (input) {
      const buffer = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        buffer[i] = Math.max(-1, Math.min(1, input[i])) * 32767;
      }
      this.port.postMessage(buffer.buffer, [buffer.buffer]);
    }
    return true;
  }
}
registerProcessor('stt-processor', STTProcessor);
