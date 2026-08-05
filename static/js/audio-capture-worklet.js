class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super(options);
        this._buffer = new Float32Array(4096);
        this._writePos = 0;
        this._active = true;

        this.port.onmessage = (evt) => {
            if (evt.data && evt.data.type === "stop") {
                this._active = false;
            }
        };
    }

    process(inputs, outputs, parameters) {
        if (!this._active) return false;

        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channel = input[0];

        for (let i = 0; i < channel.length; i++) {
            this._buffer[this._writePos++] = channel[i];

            if (this._writePos >= this._buffer.length) {
                const frame = this._buffer.slice(0);
                this.port.postMessage({ type: "frame", data: frame }, [frame.buffer]);
                this._writePos = 0;
            }
        }

        return true;
    }
}

registerProcessor("audio-capture-processor", AudioCaptureProcessor);
