import os
from typing import Optional
from app.core.config import settings
from app.services.speech.stt.base import SpeechToTextProvider
from app.services.speech.stt.faster_whisper_provider import FasterWhisperProvider

class SpeechService:
    def __init__(self) -> None:
        provider_name = settings.STT_PROVIDER.lower()
        if provider_name == "faster_whisper":
            self.provider: SpeechToTextProvider = FasterWhisperProvider()
        else:
            self.provider: SpeechToTextProvider = FasterWhisperProvider()

    @classmethod
    async def warmup(cls) -> float:
        provider = FasterWhisperProvider()
        return await FasterWhisperProvider.warmup(provider.model_size)

    async def transcribe_utterance(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> Optional[str]:
        return await self.provider.transcribe_utterance(audio_bytes, language=language, prompt=prompt)
