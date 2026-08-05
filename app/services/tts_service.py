import asyncio
import inspect
from typing import AsyncGenerator, Optional
from app.core.config import settings
from app.services.speech.tts.base import TextToSpeechProvider
from app.services.speech.tts.melotts_provider import MeloTTSProvider
from app.services.speech.tts.edge_tts_provider import EdgeTTSProvider
from app.core.logging import logger

class VoiceService:
    def __init__(self) -> None:
        from app.core.config import check_low_memory
        low_mem = check_low_memory()

        provider_name = settings.TTS_PROVIDER.lower()
        if low_mem:
            logger.info("[VoiceService] Low memory environment detected. Enforcing EdgeTTSProvider.")
            self.provider: TextToSpeechProvider = EdgeTTSProvider()
        elif provider_name == "edge_tts":
            self.provider = EdgeTTSProvider()
            logger.info("[VoiceService] Using EdgeTTSProvider.")
        else:
            try:
                candidate = MeloTTSProvider()
                self.provider = candidate
                logger.info("[VoiceService] Using MeloTTSProvider.")
            except Exception as e:
                logger.warning(f"[VoiceService] MeloTTS unavailable ({e}). Falling back to EdgeTTSProvider.")
                self.provider = EdgeTTSProvider()

        # Cache inspect.signature results at initialization
        sig = inspect.signature(self.provider.stream_speech)
        self._has_voice_config = "voice_config" in sig.parameters

    async def stream_speech(
        self,
        text: str,
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None
    ) -> AsyncGenerator[bytes, None]:
        if self._has_voice_config:
            async for chunk in self.provider.stream_speech(text, cancel_event=cancel_event, language=language, voice_config=voice_config):
                yield chunk
        else:
            async for chunk in self.provider.stream_speech(text, cancel_event=cancel_event, language=language):
                yield chunk

    async def stream_text_stream_progressive(
        self,
        text_stream: AsyncGenerator[str, None],
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None
    ) -> AsyncGenerator[bytes, None]:
        buffer = ""
        punctuation = {'.', '?', '!', '\n'}
        abbreviations = ("dr.", "mr.", "mrs.", "ms.", "vs.", "st.", "co.", "inc.", "ltd.", "e.g.", "i.e.")

        async for chunk in text_stream:
            if cancel_event and cancel_event.is_set():
                break
            buffer += chunk

            while True:
                first_idx = -1
                for p in punctuation:
                    idx = buffer.find(p)
                    if idx != -1:
                        if first_idx == -1 or idx < first_idx:
                            first_idx = idx

                if first_idx == -1:
                    break

                segment = buffer[:first_idx + 1]
                low_seg = segment.strip().lower()
                if any(low_seg.endswith(abbr) for abbr in abbreviations):
                    break

                sentence = segment.strip()
                buffer = buffer[first_idx + 1:]

                if sentence:
                    async for audio_chunk in self.stream_speech(
                        sentence,
                        cancel_event=cancel_event,
                        language=language,
                        voice_config=voice_config
                    ):
                        if cancel_event and cancel_event.is_set():
                            return
                        yield audio_chunk

        remaining = buffer.strip()
        if remaining and (not cancel_event or not cancel_event.is_set()):
            async for audio_chunk in self.stream_speech(
                remaining,
                cancel_event=cancel_event,
                language=language,
                voice_config=voice_config
            ):
                if cancel_event and cancel_event.is_set():
                    return
                yield audio_chunk
