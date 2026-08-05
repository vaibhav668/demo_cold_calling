import asyncio
import io
import audioop
from typing import AsyncGenerator, Optional
from app.core.logging import logger
from app.services.speech.tts.base import TextToSpeechProvider

# Maps (persona_name, language_code) → edge-tts voice short-name.
_VOICE_MAP: dict = {
    ("sophia", "en"):  "en-IN-NeerjaNeural",
    ("sophia", "hi"):  "hi-IN-SwaraNeural",
    ("sophia", "te"):  "te-IN-ShrutiNeural",

    ("maya", "en"):    "en-US-JennyNeural",
    ("maya", "hi"):    "hi-IN-SwaraNeural",
    ("maya", "te"):    "te-IN-ShrutiNeural",

    ("ananya", "en"):  "en-IN-NeerjaNeural",
    ("ananya", "hi"):  "hi-IN-SwaraNeural",
    ("ananya", "te"):  "te-IN-ShrutiNeural",

    ("arjun", "en"):   "en-IN-PrabhatNeural",
    ("arjun", "hi"):   "hi-IN-MadhurNeural",
    ("arjun", "te"):   "te-IN-MohanNeural",

    ("david", "en"):   "en-US-GuyNeural",
    ("david", "hi"):   "hi-IN-MadhurNeural",
    ("david", "te"):   "te-IN-MohanNeural",
}

_DEFAULT_VOICE = "en-IN-NeerjaNeural"

def _resolve_voice(voice_config: Optional[dict], language: Optional[str]) -> str:
    lang = (language or "en").split("-")[0].lower()  # 'en', 'hi', 'te'

    if voice_config:
        persona = (
            voice_config.get("persona_name") or
            voice_config.get("voice_name") or
            voice_config.get("name") or ""
        ).lower().strip()

        edge_voice = voice_config.get("edge_voice") or voice_config.get("edge_tts_voice")
        if edge_voice:
            return edge_voice

        key = (persona, lang)
        if key in _VOICE_MAP:
            return _VOICE_MAP[key]

        fallback_key = (persona, "en")
        if fallback_key in _VOICE_MAP:
            return _VOICE_MAP[fallback_key]

    lang_defaults = {
        "hi": "hi-IN-SwaraNeural",
        "te": "te-IN-ShrutiNeural",
        "en": "en-IN-NeerjaNeural",
    }
    return lang_defaults.get(lang, _DEFAULT_VOICE)

def _mp3_to_mulaw_chunks(mp3_bytes: bytes, chunk_size: int = 160) -> list:
    try:
        import miniaudio
        decoded = miniaudio.decode(
            mp3_bytes,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=8000
        )
        pcm_bytes = decoded.samples.tobytes()
        pcm_len = len(pcm_bytes)
        samples_count = pcm_len // 2
        duration_sec = samples_count / 8000.0
        
        logger.info(f"[EdgeTTS] PCM Output: sample_rate=8000, channels=1, buffer={pcm_len} bytes, duration={duration_sec:.3f}s")
        mulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
    except Exception as e:
        logger.error(f"[EdgeTTS] MP3→PCM conversion failed: {e}")
        return []

    chunks = []
    for i in range(0, len(mulaw_bytes), chunk_size):
        chunk = mulaw_bytes[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk.ljust(chunk_size, b'\xff')
        chunks.append(chunk)
    return chunks

class EdgeTTSProvider(TextToSpeechProvider):
    async def stream_speech(
        self,
        text: str,
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None,
    ) -> AsyncGenerator[bytes, None]:
        import re
        clean_text = text.strip()
        # Verify text contains at least one letter or number (supports English, Devanagari, Telugu scripts)
        if not clean_text or not re.search(r'[\w\u0900-\u097F\u0C00-\u0C7F]', clean_text):
            logger.debug(f"[EdgeTTS] Skipping punctuation-only/empty text chunk: '{text}'")
            return

        try:
            import edge_tts
        except ImportError:
            logger.error("[EdgeTTS] edge-tts package not installed. Cannot synthesize.")
            return

        voice = _resolve_voice(voice_config, language)
        logger.info(f"[EdgeTTS] Synthesizing: voice='{voice}' lang='{language}' chars={len(text)}")

        try:
            communicate = edge_tts.Communicate(text, voice)
            mp3_buffer = bytearray()
            async for chunk in communicate.stream():
                if cancel_event and cancel_event.is_set():
                    logger.info("[EdgeTTS] Cancelled mid-synthesis.")
                    return
                if chunk["type"] == "audio":
                    mp3_buffer.extend(chunk["data"])

            if not mp3_buffer:
                logger.warning("[EdgeTTS] No audio data received.")
                return

            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None, _mp3_to_mulaw_chunks, bytes(mp3_buffer)
            )

            for chunk in chunks:
                if cancel_event and cancel_event.is_set():
                    return
                yield chunk

        except Exception as e:
            logger.error(f"[EdgeTTS] Synthesis error: {e}", exc_info=True)
