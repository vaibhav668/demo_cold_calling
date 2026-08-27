import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

class TextToSpeechProvider(ABC):
    """Abstract base class representing a Text-to-Speech (TTS) provider."""

    @abstractmethod
    async def stream_speech(
        self,
        text: str,
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text into speech and yield G.711 mu-law 20ms chunks.

        Args:
            text: The text string to synthesize.
            cancel_event: Optional asyncio Event to stop execution mid-stream on barge-in.
            language: Optional hint for target speech language ('en', 'hi', 'te').

        Yields:
            160-byte (20ms) G.711 mu-law audio chunks.
        """
        yield b""

    async def stream_text_stream_progressive(
        self,
        text_generator: AsyncGenerator[str, None],
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Progressive LLM Token -> Sentence Splitter -> Sequenced Parallel Synthesis -> Ordered Playout.
        Default implementation delegates to VoiceService facade.
        """
        from app.services.tts_service import VoiceService
        vs = VoiceService(self)
        async for chunk in vs.stream_text_stream_progressive(
            text_generator,
            cancel_event=cancel_event,
            language=language,
            voice_config=voice_config
        ):
            yield chunk

