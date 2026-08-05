import audioop
from typing import Optional
from app.core.config import settings
from app.core.logging import logger
from app.services.speech.vad.base import VoiceActivityDetector
from app.services.speech.vad.silero_provider import SileroVADProvider

def _rms_ulaw(audio_chunk: bytes) -> float:
    if not audio_chunk:
        return 0.0
    pcm_bytes = audioop.ulaw2lin(audio_chunk, 2)
    return audioop.rms(pcm_bytes, 2)

class LegacyRMSDetector(VoiceActivityDetector):
    """Fallback VAD using dynamic noise floor estimation."""
    def __init__(self) -> None:
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._speech_confirmed = False
        self.noise_floor = None

    def process_frame(self, audio_chunk: bytes) -> Optional[str]:
        rms = _rms_ulaw(audio_chunk)
        if self.noise_floor is None:
            self.noise_floor = max(50.0, min(800.0, rms))

        if rms < self.noise_floor:
            self.noise_floor = 0.90 * self.noise_floor + 0.10 * rms
        else:
            if not self._in_speech:
                self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms

        self.noise_floor = max(50.0, min(800.0, self.noise_floor))

        speech_threshold = max(380.0, self.noise_floor + 250.0)
        silence_threshold = max(200.0, self.noise_floor + 100.0)

        if not self._in_speech:
            if rms > speech_threshold:
                self._speech_frames += 1
                if self._speech_frames >= 3 and not self._speech_confirmed:
                    self._in_speech = True
                    self._speech_confirmed = True
                    self._silence_frames = 0
                    return 'speech_start'
            else:
                self._speech_frames = max(0, self._speech_frames - 1)
        else:
            if rms < silence_threshold:
                self._silence_frames += 1
                if self._silence_frames >= 20:
                    self._in_speech = False
                    self._speech_frames = 0
                    self._silence_frames = 0
                    self._speech_confirmed = False
                    return 'speech_end'
            else:
                self._silence_frames = 0
        return None

    def reset(self) -> None:
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._speech_confirmed = False

class EndOfSpeechDetector:
    def __init__(self) -> None:
        provider_name = settings.VAD_PROVIDER.lower()
        self.provider: VoiceActivityDetector = None
        self._fallback_provider = LegacyRMSDetector()

        if provider_name == "silero":
            self.provider = SileroVADProvider()
            if self.provider.model is None or self.provider.model == "FAILED":
                logger.warning("[VAD] Silero VAD failed. Falling back to dynamic RMS VAD.")
                self.provider = self._fallback_provider
        else:
            self.provider = self._fallback_provider

    def process_frame(self, audio_chunk: bytes) -> Optional[str]:
        return self.provider.process_frame(audio_chunk)

    def reset(self) -> None:
        self.provider.reset()

    @property
    def is_speaking(self) -> bool:
        if isinstance(self.provider, SileroVADProvider):
            return self.provider.is_speaking
        return self._fallback_provider._in_speech
