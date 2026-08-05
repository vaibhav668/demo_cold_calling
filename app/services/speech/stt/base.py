from abc import ABC, abstractmethod
from typing import Optional

class SpeechToTextProvider(ABC):
    """Abstract base class representing a Speech-to-Text provider."""

    @abstractmethod
    async def transcribe_utterance(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None
    ) -> Optional[str]:
        """
        Transcribes the given raw G.711 mu-law audio bytes into text.

        Args:
            audio_bytes: The G.711 mu-law audio payload.
            language: Optional hint for transcription language.

        Returns:
            The transcribed text, or None if transcription failed or was silent.
        """
        pass

    def clear_buffer(self) -> None:
        """Resets any internal audio buffer, if applicable."""
        pass
