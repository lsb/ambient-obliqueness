class MeterProcessor extends AudioWorkletProcessor {
    constructor () {
      super();
      this.buffer = [];
      this.window = Math.round(sampleRate * 0.5);      // global in worklet scope
    }
  
    process (inputs) {
      const input = inputs[0];
      if (!input.length) return true;
  
      // Mono: just look at channel 0. (For stereo, loop channels.)
      const channel = input[0];
      this.buffer.push(...channel);
  
      if (this.buffer.length >= this.window) {          // ≈ 1 second collected
        const frame = this.buffer.splice(0, this.window);
  
        // Peak
        let max = 0;
        for (const v of frame) {
          const a = Math.abs(v);
          if (a > max) max = a;
        }
  
        // Std dev
        let sum = 0;
        for (const v of frame) sum += v;
        const mean = sum / frame.length;
  
        let variance = 0;
        for (const v of frame) {
          const d = v - mean;
          variance += d * d;
        }
        const sd = Math.sqrt(variance / frame.length);
  
        // Send to main thread
        this.port.postMessage({ max, sd, frame });
      }
      return true;   // keep processor alive
    }
}
  
registerProcessor('meter', MeterProcessor);
  