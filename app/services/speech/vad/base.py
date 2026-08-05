from abc import ABC, abstractmethod
from typing import Optional

class VoiceActivityDetector(ABC):
    """Abstract base class representing a Voice Activity Detector (VAD)."""

    @abstractmethod
    def process_frame(self, audio_chunk: bytes) -> Optional[str]:
        """
        Process a 20ms G.711 mu-law audio frame.

        Args:
            audio_chunk: G.711 mu-law raw audio chunk (typically 160 bytes for 20ms at 8kHz).

        Returns:
            'speech_start'  - speech start boundary detected
            'speech_end'    - speech end boundary detected
            None            - no event detected in this frame
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal voice activity state."""
        pass
