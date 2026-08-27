import audioop
import asyncio
import numpy as np
import collections
from typing import Optional
from app.core.logging import logger
from app.services.speech.vad.base import VoiceActivityDetector

def _pcm16_chunk_to_float32_16k(audio_chunk: bytes) -> np.ndarray:
    """
    Convert a 16kHz 16-bit PCM (pcm_s16le) chunk → float32 numpy array.
    If chunk length is odd, trim the last byte.
    """
    if len(audio_chunk) % 2 != 0:
        audio_chunk = audio_chunk[:-1]
    return np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

class SileroVADProvider(VoiceActivityDetector):
    """
    VAD provider utilizing Silero VAD.
    """

    _model_instance = None

    def __init__(self) -> None:
        self.model = None
        self.vad_iterator = None
        self._in_speech = False
        self._np_accumulator = np.empty(0, dtype=np.float32)

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
                    threshold=0.4,                  # Lowered from 0.5 to detect speech onset faster
                    sampling_rate=16000,
                    min_silence_duration_ms=400,    # 400ms silence to finalize speech segment
                    speech_pad_ms=30
                )
                logger.info("[VAD] Silero VAD iterator initialized.")
        except Exception as e:
            logger.error(f"[VAD] Failed to initialize Silero VAD model: {e}. Fallback enabled.")
            SileroVADProvider._model_instance = "FAILED"

    def process_frame(self, audio_chunk: bytes) -> Optional[str]:
        if not audio_chunk or self.model is None or self.vad_iterator is None:
            return None

        x_16k = _pcm16_chunk_to_float32_16k(audio_chunk)
        self._np_accumulator = np.concatenate([self._np_accumulator, x_16k])
        detected_event = None

        while len(self._np_accumulator) >= 512:
            block = self._np_accumulator[:512]
            self._np_accumulator = self._np_accumulator[512:]

            try:
                import torch
                block_tensor = torch.from_numpy(block)
                with torch.inference_mode():
                    result = self.vad_iterator(block_tensor)

                if result:
                    if "start" in result:
                        self._in_speech = True
                        if detected_event is None:
                            detected_event = "speech_start"
                    elif "end" in result:
                        self._in_speech = False
                        detected_event = "speech_end"
            except Exception as e:
                logger.warning(f"[VAD] Silero frame iteration error: {e}")

        return detected_event

    def reset(self) -> None:
        self._np_accumulator = np.empty(0, dtype=np.float32)
        self._in_speech = False
        if self.vad_iterator is not None:
            try:
                if hasattr(self.vad_iterator, "reset_states"):
                    self.vad_iterator.reset_states()
                else:
                    self.vad_iterator.reset()
            except Exception:
                pass

    @property
    def is_speaking(self) -> bool:
        return self._in_speech
