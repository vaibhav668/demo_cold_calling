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
