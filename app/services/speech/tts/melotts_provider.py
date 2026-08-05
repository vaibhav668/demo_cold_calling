import os
import asyncio
from typing import AsyncGenerator, Optional
from app.core.logging import logger
from app.services.speech.tts.base import TextToSpeechProvider

class MeloTTSProvider(TextToSpeechProvider):
    """
    Stub wrapper for MeloTTS to maintain class hierarchy.
    """
    async def stream_speech(
        self,
        text: str,
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None
    ) -> AsyncGenerator[bytes, None]:
        # Under low memory conditions in demo, fallback directly to EdgeTTS
        logger.info("[MeloTTS] Running under low memory configuration. Redirecting synthesis to EdgeTTS.")
        from app.services.speech.tts.edge_tts_provider import EdgeTTSProvider
        provider = EdgeTTSProvider()
        async for chunk in provider.stream_speech(text, cancel_event, language, voice_config):
            yield chunk
