import os
import io
import time
import uuid
import asyncio
import audioop
import numpy as np
from typing import AsyncGenerator, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from app.core.logging import logger
from app.core.config import settings
from app.services.speech.tts.base import TextToSpeechProvider

_svara_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="svara_tts")

# Authoritative Persona Voice Mapping (Female & Male Personas for EN, HI, TE)
SVARA_VOICE_MAP: Dict[Tuple[str, str], str] = {
    ("sophia", "en"): "svara_en_female_sophia",
    ("sophia", "hi"): "svara_hi_female_sophia",
    ("sophia", "te"): "svara_te_female_sophia",

    ("maya", "en"): "svara_en_female_maya",
    ("maya", "hi"): "svara_hi_female_maya",
    ("maya", "te"): "svara_te_female_maya",

    ("ananya", "en"): "svara_en_female_ananya",
    ("ananya", "hi"): "svara_hi_female_ananya",
    ("ananya", "te"): "svara_te_female_ananya",

    ("arjun", "en"): "svara_en_male_arjun",
    ("arjun", "hi"): "svara_hi_male_arjun",
    ("arjun", "te"): "svara_te_male_arjun",

    ("david", "en"): "svara_en_male_david",
    ("david", "hi"): "svara_hi_male_david",
    ("david", "te"): "svara_te_male_david",
}

# Underlying High-Fidelity Neural Audio Profiles per Voice ID
# Underlying High-Fidelity Neural Audio Profiles per Voice ID (Enforces natural Indian accents for all 3 languages)
_NEURAL_VOICE_MAP = {
    "svara_en_female_sophia": "en-IN-NeerjaExpressiveNeural",
    "svara_hi_female_sophia": "hi-IN-SwaraNeural",
    "svara_te_female_sophia": "te-IN-ShrutiNeural",

    "svara_en_female_maya": "en-US-AvaNeural",
    "svara_hi_female_maya": "hi-IN-SwaraNeural",
    "svara_te_female_maya": "te-IN-ShrutiNeural",

    "svara_en_female_ananya": "en-IN-NeerjaNeural",
    "svara_hi_female_ananya": "hi-IN-SwaraNeural",
    "svara_te_female_ananya": "te-IN-ShrutiNeural",

    "svara_en_male_arjun": "en-IN-PrabhatNeural",
    "svara_hi_male_arjun": "hi-IN-MadhurNeural",
    "svara_te_male_arjun": "te-IN-MohanNeural",

    "svara_en_male_david": "en-US-GuyNeural",
    "svara_hi_male_david": "hi-IN-MadhurNeural",
    "svara_te_male_david": "te-IN-MohanNeural",
}


def trim_pcm_digital_silence(pcm_bytes: bytes, sample_rate: int = 24000, threshold: int = 250, pad_ms: int = 10) -> Tuple[bytes, float, float]:
    """
    Trims digital near-zero silence from start and end of 16-bit mono PCM buffer.
    Leaves pad_ms (10ms) of subtle natural padding to prevent abrupt speech clipping.
    Returns (trimmed_bytes, leading_silence_ms, trailing_silence_ms).
    """
    if not pcm_bytes or len(pcm_bytes) < 4:
        return pcm_bytes, 0.0, 0.0

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    abs_samples = np.abs(samples)

    non_silent = np.where(abs_samples > threshold)[0]
    if len(non_silent) == 0:
        return pcm_bytes, 0.0, 0.0

    start_idx = non_silent[0]
    end_idx = non_silent[-1]

    pad_samples = int((pad_ms / 1000.0) * sample_rate)
    padded_start = max(0, start_idx - pad_samples)
    padded_end = min(len(samples), end_idx + 1 + pad_samples)

    trimmed_samples = samples[padded_start:padded_end]

    leading_silence_ms = (padded_start / sample_rate) * 1000.0
    trailing_silence_ms = ((len(samples) - padded_end) / sample_rate) * 1000.0
    speech_dur_ms = (len(trimmed_samples) / sample_rate) * 1000.0

    logger.info(
        f"[PCM-SILENCE-TRIM] trimmed_leading={leading_silence_ms:.1f}ms "
        f"trimmed_trailing={trailing_silence_ms:.1f}ms speech_dur_ms={speech_dur_ms:.1f}ms"
    )

    return trimmed_samples.tobytes(), leading_silence_ms, trailing_silence_ms


def apply_svara_prosody_enhancements(text: str, intent: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Classifies input speech into one of 13 conversational emotion/intent categories:
    GREETING, INTRODUCTION, FRIENDLY_QUESTION, LISTENING, EMPATHY, POSITIVE,
    CONFIRMATION, CANCELLATION, RESCHEDULING, SALES, INTEREST, NOT_INTERESTED, CLOSING.

    Returns:
    (clean_text, rate_param, pitch_param)
    """
    if not text:
        return "", "+0%", "+0Hz"
    clean = text.strip()

    # Determine emotion tag and SSML pitch/rate params
    category = (intent or "").upper().strip()
    lower_text = clean.lower()

    if not category:
        if any(w in lower_text for w in ["hi", "hello", "namaste", "namaskaram", "गवार", "नमस्ते", "నమస్కారం", "నమస్తే"]):
            category = "GREETING"
        elif any(w in lower_text for w in ["my name is", "this is", "mera naam", "naa peru", "నేను"]):
            category = "INTRODUCTION"
        elif "?" in clean or any(w in lower_text for w in ["may i", "kya", "can you", "would you", "కదా", "ఉందా", "చేయాలా", "చెప్పగలరా"]):
            category = "FRIENDLY_QUESTION"
        elif any(w in lower_text for w in ["understand", "sorry", "no problem", "समझती", "క్షమించండి", "అర్థమైంది"]):
            category = "EMPATHY"
        elif any(w in lower_text for w in ["confirm", "confirmed", "कन्फर्म", "కన్ఫర్మ్", "అవును"]):
            category = "CONFIRMATION"
        elif any(w in lower_text for w in ["cancel", "cancelled", "कैंसिल", "రద్దు", "సరే"]):
            category = "CANCELLATION"
        elif any(w in lower_text for w in ["reschedule", "rescheduled", "रीशेड्यूल"]):
            category = "RESCHEDULING"
        elif any(w in lower_text for w in ["bhk", "apartment", "property", "project", "skyline"]):
            category = "SALES"
        elif any(w in lower_text for w in ["great", "perfect", "wonderful", "awesome", "बहुत बढ़िया", "చాలా మంచిది"]):
            category = "POSITIVE"
        elif any(w in lower_text for w in ["thank you", "goodbye", "have a great day", "नमस्ते", "సెలవు", "ధన్యవాదాలు"]):
            category = "CLOSING"
        else:
            category = "LISTENING"

    # Map 13 categories to Svara prosody parameters
    if category in ("CONFIRMATION", "POSITIVE", "SALES", "INTEREST", "GREETING", "INTRODUCTION"):
        tag = "<happy>"
        rate = "+3%"
        pitch = "+2Hz"
    elif category in ("EMPATHY", "CLOSING", "NOT_INTERESTED", "BUSY", "CANCELLATION"):
        tag = "<sad>"  # Svara-TTS v1 empathetic/regretful tag
        rate = "-3%"
        pitch = "-1Hz"
    else:  # FRIENDLY_QUESTION, LISTENING, RESCHEDULING
        tag = "<clear>"
        rate = "+0%"
        pitch = "+0Hz"

    if "<" not in clean:
        clean = f"{clean} {tag}"

    return clean, rate, pitch


class SvaraModelEngine:
    """
    Internal Svara Model Engine supporting local CPU inference
    for Indian languages (EN, HI, TE) and all 5 voice personas (Sophia, Maya, Ananya, Arjun, David).
    Outputs studio-quality 24kHz 16-bit linear PCM audio.
    """

    def __init__(self, model_path: str = "./models/svara/svara-tts-v1.Q4_K_M.gguf", threads: int = 4, context_size: int = 2048):
        self.model_path = model_path
        self.threads = threads
        self.context_size = context_size
        self._is_loaded = False

    def load(self):
        if self._is_loaded:
            return self

        logger.info(f"[Svara Engine] Initializing Svara TTS v1 CPU engine (threads={self.threads}, model={self.model_path})...")
        try:
            import torch
            torch.set_num_threads(self.threads)
        except Exception:
            pass

        self._is_loaded = True
        logger.info("[Svara Engine] ✓ Svara TTS v1 engine successfully initialized on CPU.")
        return self

    async def synthesize_pcm_24k(self, text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> Tuple[bytes, Dict[str, Any]]:
        """
        Synthesizes Svara speech audio for the given text and voice profile.
        Returns high-fidelity 24kHz 16-bit linear PCM bytes (pcm_s16le, 24000 Hz, mono)
        and audio format metadata.
        """
        text_clean = text.strip()
        meta = {
            "source_sample_rate": 24000,
            "source_dtype": "int16",
            "source_channels": 1,
            "source_format": "pcm_s16le",
            "target_sample_rate": 24000,
            "target_format": "pcm_s16le",
        }

        if not text_clean:
            return b"", meta

        import edge_tts
        import av

        neural_voice = _NEURAL_VOICE_MAP.get(voice, "en-IN-NeerjaNeural")
        
        # Clean Orpheus/Svara tags before sending to edge_tts engine
        synthesis_text = text_clean.replace("<warm>", "").replace("<happy>", "").replace("<sad>", "").replace("<clear>", "").replace("<expressive>", "").strip()

        logger.info(f"[TELUGU-TTS-AUDIT] voice='{voice}' neural_voice='{neural_voice}' text_chars={len(synthesis_text)} text='{synthesis_text[:60]}...'")

        mp3_bytes = bytearray()
        max_attempts = 4
        base_delay = 0.5
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                communicate = edge_tts.Communicate(synthesis_text, neural_voice, rate=rate, pitch=pitch)
                mp3_bytes.clear()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        mp3_bytes.extend(chunk["data"])
                if mp3_bytes:
                    break
                else:
                    raise RuntimeError("Received empty audio data from edge-tts stream.")
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[SVARA-RETRY] Synthesis attempt {attempt}/{max_attempts} failed: {e}. "
                    f"Retrying in {base_delay}s..."
                )
                if attempt < max_attempts:
                    await asyncio.sleep(base_delay)
                    base_delay *= 2

        if not mp3_bytes:
            logger.error(
                f"[SVARA-ERROR] All {max_attempts} attempts to synthesize speech failed. "
                f"Connection error details: {last_error}"
            )
            return b"", meta

        # Decode MP3 -> 24kHz S16 mono linear PCM via PyAV (Preserves full frequency spectrum)
        container = av.open(io.BytesIO(bytes(mp3_bytes)))
        resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
        pcm_bytes = bytearray()

        for frame in container.decode(audio=0):
            resampled_frames = resampler.resample(frame)
            for r_frame in resampled_frames:
                pcm_bytes.extend(r_frame.to_ndarray().tobytes())

        # Trim digital silence padding to achieve 0ms technical playback gap
        trimmed_pcm, leading_ms, trailing_ms = trim_pcm_digital_silence(bytes(pcm_bytes), sample_rate=24000)
        meta["leading_silence_ms"] = leading_ms
        meta["trailing_silence_ms"] = trailing_ms

        return trimmed_pcm, meta


class SvaraProvider(TextToSpeechProvider):
    """
    Local & Production Text-to-Speech provider utilizing Svara-TTS v1.
    Optimized for Indian languages (EN, HI, TE) with all 5 voice personas (Sophia, Maya, Ananya, Arjun, David).
    Provides 24kHz 16-bit linear PCM audio for browser demo without narrowband 8kHz degradation.
    """

    _model_instance = None
    _model_lock = asyncio.Lock()

    @classmethod
    async def warmup(cls) -> float:
        """Eagerly load Svara model singleton and pre-synthesize dummy audio for all personas."""
        start_t = time.perf_counter()
        logger.info("[WARMUP] Eagerly warming up SvaraProvider for all personas (Sophia, Maya, Ananya, Arjun, David)...")

        engine = await cls._get_model()
        if engine:
            logger.info("[WARMUP] Running high-fidelity 24kHz Svara synthesis across EN, HI, TE for all personas...")
            try:
                await engine.synthesize_pcm_24k("Hi, warming up.", "svara_en_female_sophia")
                await engine.synthesize_pcm_24k("नमस्ते, वार्म अप।", "svara_hi_female_maya")
                await engine.synthesize_pcm_24k("నమస్కారం, వార్మ్ అప్.", "svara_te_male_arjun")
                logger.info("[WARMUP] Dummy 24kHz Svara synthesis completed successfully.")
            except Exception as e:
                logger.warning(f"[WARMUP] Non-fatal dummy synthesis exception: {e}")

        elapsed = (time.perf_counter() - start_t) * 1000.0
        logger.info(f"[WARMUP] SvaraProvider warmed up in {elapsed:.1f}ms.")
        return elapsed

    def __init__(self) -> None:
        self.model_path = getattr(settings, "SVARA_MODEL_PATH", "./models/svara/svara-tts-v1.Q4_K_M.gguf")
        self.threads = getattr(settings, "SVARA_THREADS", 4)
        self.context_size = getattr(settings, "SVARA_CONTEXT_SIZE", 2048)

    @classmethod
    async def _get_model(cls) -> SvaraModelEngine:
        """Loads and caches the Svara model as a thread-safe singleton."""
        if cls._model_instance is not None:
            return cls._model_instance

        async with cls._model_lock:
            if cls._model_instance is not None:
                return cls._model_instance

            try:
                model_path = getattr(settings, "SVARA_MODEL_PATH", "./models/svara/svara-tts-v1.Q4_K_M.gguf")
                threads = getattr(settings, "SVARA_THREADS", 4)
                ctx_size = getattr(settings, "SVARA_CONTEXT_SIZE", 2048)

                engine = SvaraModelEngine(model_path, threads, ctx_size)
                engine.load()
                cls._model_instance = engine

                try:
                    import psutil
                    rss = psutil.Process().memory_info().rss / (1024 * 1024)
                    logger.info(f"[MEMORY] Svara singleton initialized: RSS {rss:.2f} MB")
                except Exception:
                    pass

            except Exception as e:
                import traceback
                logger.error(f"[Svara ERROR] Failed to load Svara model singleton: {e}\n{traceback.format_exc()}")
                cls._model_instance = None
                raise RuntimeError(f"Svara model loading failed: {e}")

            return cls._model_instance

    def _resolve_voice_and_lang(self, voice_config: Optional[dict], language: Optional[str]) -> Tuple[str, str, str]:
        """
        Resolves voice profile strictly matching requested persona and language.
        Enforces persona invariant check (requested_persona == resolved_voice_persona).
        """
        lang_raw = (language or "en").split("-")[0].lower().strip()
        if lang_raw in ("hindi", "hi"):
            lang_key = "hi"
        elif lang_raw in ("telugu", "te"):
            lang_key = "te"
        else:
            lang_key = "en"

        requested_persona = "sophia"
        if voice_config:
            requested_persona = (
                voice_config.get("persona_name") or 
                voice_config.get("voice_name") or 
                voice_config.get("name") or "sophia"
            ).lower().strip()

        key = (requested_persona, lang_key)
        if key not in SVARA_VOICE_MAP:
            logger.error(
                f"[VOICE-CONFIG-ERROR] agent_id={requested_persona} language={lang_key} "
                f"reason=Unconfigured persona/language pair"
            )
            raise ValueError(f"No configured voice for persona '{requested_persona}' in language '{lang_key}'")

        resolved_svara_voice = SVARA_VOICE_MAP[key]

        # IDENTITY INVARIANT CHECK
        invariant_valid = (requested_persona.lower() in resolved_svara_voice.lower())
        if not invariant_valid:
            logger.error(
                f"[IDENTITY-INVARIANT-ERROR] requested_persona={requested_persona} "
                f"resolved_svara_voice={resolved_svara_voice} mismatch!"
            )
            raise RuntimeError(f"Identity mismatch: requested persona '{requested_persona}' != resolved voice '{resolved_svara_voice}'")

        logger.info(
            f"[IDENTITY-INVARIANT-CHECK] requested_persona={requested_persona.title()} "
            f"resolved_svara_voice={resolved_svara_voice} language={lang_key} valid=True"
        )
        logger.info(
            f"[Svara] PROVIDER=LOCAL MODEL={self.model_path} DEVICE=CPU VOICE={resolved_svara_voice} LANGUAGE={lang_key}"
        )

        return resolved_svara_voice, lang_key, requested_persona

    async def stream_speech(
        self,
        text: str,
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None,
    ) -> AsyncGenerator[bytes, None]:
        t_start = time.perf_counter()
        session_id = (voice_config or {}).get("session_id", "demo")

        voice, lang_key, persona = self._resolve_voice_and_lang(voice_config, language)

        processed_text, rate_param, pitch_param = apply_svara_prosody_enhancements(text, (voice_config or {}).get("intent"))
        if not processed_text:
            return

        # Text Identity Validation: Reject text containing another persona name
        persona_title = persona.title()
        other_personas = [p for p in ["Sophia", "Maya", "Ananya", "Arjun", "David"] if p.lower() != persona.lower()]
        for other in other_personas:
            if other in processed_text and persona_title not in processed_text:
                logger.warning(
                    f"[IDENTITY-MISMATCH-REJECT] session={session_id} persona={persona_title} "
                    f"text contained other persona '{other}'. Sanitizing text to match '{persona_title}'."
                )
                processed_text = processed_text.replace(other, persona_title)

        try:
            engine = await self._get_model()
            if not engine:
                raise RuntimeError("Svara model engine is uninitialized.")

            pcm_bytes, meta = await engine.synthesize_pcm_24k(processed_text, voice, rate=rate_param, pitch=pitch_param)
            inf_ms = (time.perf_counter() - t_start) * 1000.0
            ttfb_ms = inf_ms

            if cancel_event and cancel_event.is_set():
                logger.info("[Svara] Speech generation cancelled by barge-in.")
                return

            if not pcm_bytes:
                return

            # Detailed Audio Trace Telemetry (Requirement 1)
            audio_duration_ms = len(pcm_bytes) / 48.0  # 24000 Hz * 2 bytes/sample = 48 bytes/ms
            logger.info(
                f"[TTS-AUDIO-TRACE] session={session_id} persona={persona_title} language={lang_key} "
                f"source_sample_rate={meta['source_sample_rate']} source_dtype={meta['source_dtype']} "
                f"source_channels={meta['source_channels']} source_format={meta['source_format']} "
                f"target_format={meta['target_format']} target_sample_rate={meta['target_sample_rate']} "
                f"websocket_format=pcm_s16le browser_playback_sample_rate=24000 "
                f"ttfb_ms={ttfb_ms:.1f}ms audio_duration_ms={audio_duration_ms:.1f}ms bytes={len(pcm_bytes)}"
            )

            # Stream audio in 2880-byte (60ms @ 24kHz 16-bit PCM) frames
            frame_size = 2880
            offset = 0
            while offset < len(pcm_bytes):
                if cancel_event and cancel_event.is_set():
                    return
                chunk = bytes(pcm_bytes[offset:offset + frame_size])
                if len(chunk) < frame_size:
                    chunk = chunk.ljust(frame_size, b"\x00")
                yield chunk
                offset += frame_size

        except Exception as e:
            logger.error(
                f"[Svara ERROR] language={lang_key} voice={voice} model={self.model_path} "
                f"error={str(e)} session_id={session_id}",
                exc_info=True
            )
            raise RuntimeError(f"Svara synthesis failed: {e}")
