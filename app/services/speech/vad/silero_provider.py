import audioop
import asyncio
import numpy as np
import collections
from typing import Optional
from app.core.logging import logger
from app.services.speech.vad.base import VoiceActivityDetector

def _ulaw_chunk_to_float32_16k(audio_chunk: bytes) -> np.ndarray:
    """
    Convert a G.711 mu-law 8kHz chunk → float32 numpy array at 16kHz.
    Uses C-level audioop.
    """
    pcm_8k = audioop.ulaw2lin(audio_chunk, 2)
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
    return np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0

class SileroVADProvider(VoiceActivityDetector):
    """
    VAD provider utilizing Silero VAD.
    """

    _model_instance = None

    def __init__(self) -> None:
        self.model = None
        self.vad_iterator = None
        self._accumulator = collections.deque()
        self._in_speech = False

        try:
            from silero_vad import load_silero_vad, VADIterator
            if SileroVADProvider._model_instance is None:
                logger.info("[VAD] Loading Silero VAD model...")
                SileroVADProvider._model_instance = load_silero_vad()
                try:
                    import psutil
                    rss = psutil.Process().memory_info().rss / (1024 * 1024)
                    logger.info(f"[MEMORY] Silero loaded: RSS {rss:.2f} MB")
                except Exception:
                    pass

            self.model = SileroVADProvider._model_instance
            if self.model is not None and self.model != "FAILED":
                self.vad_iterator = VADIterator(
                    self.model,
                    threshold=0.5,
                    sampling_rate=16000,
                    min_silence_duration_ms=400,
                    speech_pad_ms=30
                )
                logger.info("[VAD] Silero VAD iterator initialized.")
        except Exception as e:
            logger.error(f"[VAD] Failed to initialize Silero VAD model: {e}. Fallback enabled.")
            SileroVADProvider._model_instance = "FAILED"

    def process_frame(self, audio_chunk: bytes) -> Optional[str]:
        if not audio_chunk or self.model is None or self.vad_iterator is None:
            return None

        x_16k = _ulaw_chunk_to_float32_16k(audio_chunk)
        self._accumulator.extend(x_16k.tolist())
        event = None

        while len(self._accumulator) >= 512:
            block = []
            for _ in range(512):
                block.append(self._accumulator.popleft())

            try:
                import torch
                block_tensor = torch.tensor(block, dtype=torch.float32)
                with torch.inference_mode():
                    result = self.vad_iterator(block_tensor)

                if result:
                    if "start" in result:
                        self._in_speech = True
                        event = "speech_start"
                    elif "end" in result:
                        self._in_speech = False
                        event = "speech_end"
            except Exception as e:
                logger.warning(f"[VAD] Silero frame iteration error: {e}")

        return event

    def reset(self) -> None:
        self._accumulator.clear()
        self._in_speech = False
        if self.vad_iterator is not None:
            try:
                self.vad_iterator.reset()
            except Exception:
                pass

    @property
    def is_speaking(self) -> bool:
        return self._in_speech
