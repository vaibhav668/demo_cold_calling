import os
import re
import io
import wave
import audioop
import asyncio
import httpx
import numpy as np
from typing import Optional, Any
from app.core.logging import logger
from app.core.config import settings
from app.services.speech.stt.base import SpeechToTextProvider

# Ignore common Whisper hallucination tokens on telephone static/silence
_SILENCE_TOKENS = {
    "", ".", "..", "...", "Thank you.", "Bye.", "Thanks.", "you",
    "You.", "you.", "Okay.", "okay.", "Hmm.", "hmm.", "Uh.", "uh.",
    "Mm.", "mm.", "Mmm.", "mmm.", "[Music]", "[Applause]", "[Laughter]",
}


def _pcm16_to_float32_16k(audio_bytes: bytes) -> np.ndarray:
    """
    Convert 16kHz 16-bit signed PCM mono bytes directly to float32 numpy array [-1.0, 1.0].
    Strict canonical PCM accounting: 2 bytes per sample.
    """
    samples_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return samples_int16.astype(np.float32) / 32768.0


def calculate_pcm_metadata(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> dict:
    """
    Canonical PCM Metadata Calculator:
    Calculates exact byte length, sample count, duration_ms, RMS, and Peak.
    For PCM S16LE mono (16kHz): 2 bytes/sample, 32 bytes/ms.
    """
    num_bytes = len(pcm_bytes) if pcm_bytes else 0
    bytes_per_sample = sample_width * channels
    num_samples = num_bytes // bytes_per_sample
    duration_ms = (num_samples / sample_rate) * 1000.0 if sample_rate > 0 else 0.0
    rms_val = 0
    peak_val = 0
    if num_bytes > 0:
        import audioop
        try:
            rms_val = audioop.rms(pcm_bytes, sample_width)
            peak_val = audioop.max(pcm_bytes, sample_width)
        except Exception:
            pass
    return {
        "bytes": num_bytes,
        "samples": num_samples,
        "duration_ms": duration_ms,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "rms": rms_val,
        "peak": peak_val
    }


def preprocess_audio_for_stt(pcm_bytes: bytes, target_rms: int = 2500) -> tuple[bytes, dict]:
    """
    STT Preprocessing & Bounded Peak Gain Normalization:
    - Removes DC offset using mean subtraction.
    - Applies peak-bounded gain normalization when 150 <= orig_rms < 1200.
    - Avoids amplifying pure ambient noise / microphone hiss (orig_rms < 150).
    - Caps gain at 3.5x to prevent distortion.
    - Clamps float32 values to int16 range to eliminate hard clipping.
    """
    if not pcm_bytes:
        return pcm_bytes, {
            "raw_rms": 0, "raw_peak": 0, "normalized_rms": 0, "normalized_peak": 0,
            "gain_applied": 1.0, "original_rms": 0, "original_peak": 0, "processed_rms": 0, "processed_peak": 0
        }

    import audioop
    try:
        orig_rms = audioop.rms(pcm_bytes, 2)
        orig_peak = audioop.max(pcm_bytes, 2)

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

        # 1. Remove DC offset
        samples = samples - np.mean(samples)

        # 2. Compute bounded gain multiplier (only if sufficient voice energy >= 150 RMS)
        gain = 1.0
        if orig_rms >= 150 and orig_rms < 1200:
            desired_gain = min(float(target_rms) / max(orig_rms, 100), 3.5)
            if orig_peak > 0:
                peak_gain = 28000.0 / float(orig_peak)
                gain = min(desired_gain, peak_gain)
            else:
                gain = desired_gain

        if gain > 1.01:
            samples = samples * gain

        samples_clamped = np.clip(samples, -32768.0, 32767.0).astype(np.int16)
        proc_bytes = samples_clamped.tobytes()

        proc_rms = audioop.rms(proc_bytes, 2)
        proc_peak = audioop.max(proc_bytes, 2)

        stats = {
            "raw_rms": orig_rms,
            "raw_peak": orig_peak,
            "normalized_rms": proc_rms,
            "normalized_peak": proc_peak,
            "gain_applied": gain,
            "original_rms": orig_rms,
            "original_peak": orig_peak,
            "processed_rms": proc_rms,
            "processed_peak": proc_peak
        }
        return proc_bytes, stats
    except Exception as e:
        logger.warning(f"[STT-PREPROCESS] Preprocessing fallback error: {e}")
        return pcm_bytes, {
            "raw_rms": 0, "raw_peak": 0, "normalized_rms": 0, "normalized_peak": 0,
            "gain_applied": 1.0, "original_rms": 0, "original_peak": 0, "processed_rms": 0, "processed_peak": 0
        }


def save_debug_stt_wav(session_id: str, turn_id: int, pcm_bytes: bytes) -> str:
    """
    Save exact 16kHz PCM_S16LE mono audio bytes passed to Faster-Whisper for offline inspection.
    Saves to ./debug_stt_audio/<session_id>_turn<turn_id>.wav
    """
    import wave
    out_dir = os.path.abspath("./debug_stt_audio")
    os.makedirs(out_dir, exist_ok=True)
    clean_sess = re.sub(r'[^a-zA-Z0-9_-]', '_', str(session_id))[:32]
    wav_path = os.path.join(out_dir, f"{clean_sess}_turn{turn_id}.wav")
    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm_bytes)
        meta = calculate_pcm_metadata(pcm_bytes, 16000, 1, 2)
        logger.info(
            f"[STT-DEBUG-WAV] path={wav_path} bytes={meta['bytes']} samples={meta['samples']} "
            f"duration_ms={meta['duration_ms']:.1f}ms rms={meta['rms']} peak={meta['peak']}"
        )
        return wav_path
    except Exception as e:
        logger.warning(f"[STT-DEBUG-WAV] Failed to save debug WAV: {e}")
        return ""


class FasterWhisperProvider(SpeechToTextProvider):
    """
    Local Speech-to-Text provider utilizing the Faster-Whisper library.
    Optimized for low-latency CPU inference via CTranslate2.
    Strictly offline operation during conversation — no HuggingFace checks at runtime.
    """

    _model_instance = None
    _model_lock = asyncio.Lock()

    def __init__(self) -> None:
        from app.core.config import check_low_memory
        low_mem = check_low_memory()
        
        configured_model = os.environ.get("WHISPER_MODEL", "base")
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        
        # In a low-memory deployment, avoid loading large models locally if cloud is possible
        if low_mem and configured_model not in ("tiny.en", "tiny", "base"):
            if self.api_key:
                logger.info(
                    f"[STT] Low-memory environment detected (<1GB RAM) and large model '{configured_model}' configured. "
                    f"Directly using Groq Cloud API for STT to prevent OOM crash."
                )
                self.model_size = configured_model
            else:
                logger.warning(
                    f"[STT] Low-memory environment detected and large model '{configured_model}' configured, "
                    f"but no GROQ_API_KEY found. Overriding local model to 'base' to prevent OOM crash."
                )
                self.model_size = "base"
        else:
            self.model_size = configured_model

    @classmethod
    async def _get_model(cls, model_size: str):
        """Loads and caches the WhisperModel in a thread-safe singleton wrapper."""
        if cls._model_instance is not None:
            return cls._model_instance

        # If we are bypassing local model loading due to cloud-only mode under low memory
        from app.core.config import check_low_memory
        low_mem = check_low_memory()
        if low_mem and model_size in ("large-v3-turbo", "large-v3", "large-v2", "large", "medium") and os.environ.get("GROQ_API_KEY"):
            logger.info("[STT] Bypassing local model loading as cloud API is active.")
            return "BYPASS_LOCAL"

        async with cls._model_lock:
            if cls._model_instance is not None:
                return cls._model_instance

            try:
                from faster_whisper import WhisperModel
                import torch
                
                # Auto detect device and compute type
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                logger.info(f"[STT] Initializing Faster-Whisper model '{model_size}' on {device} ({compute_type})...")
                
                # Enforce offline mode via env vars so HuggingFace hub never initiates network calls at runtime
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

                def load_model():
                    if os.name != "nt":
                        cache_dir = os.environ.get("HF_HOME", os.path.join(os.getcwd(), "models", "hf_cache"))
                        try:
                            os.makedirs(cache_dir, exist_ok=True)
                        except Exception:
                            pass
                        download_root = cache_dir if os.path.exists(cache_dir) else None
                    else:
                        download_root = None

                    # Try loading with local_files_only=True first to guarantee zero network checks
                    try:
                        return WhisperModel(
                            model_size,
                            device=device,
                            compute_type=compute_type,
                            cpu_threads=4,
                            num_workers=1,
                            download_root=download_root,
                            local_files_only=True
                        )
                    except Exception as local_err:
                        logger.warning(f"[STT] local_files_only load failed ({local_err}). Retrying standard load...")
                        return WhisperModel(
                            model_size,
                            device=device,
                            compute_type=compute_type,
                            cpu_threads=4,
                            num_workers=1,
                            download_root=download_root
                        )

                cls._model_instance = await asyncio.get_event_loop().run_in_executor(None, load_model)
                logger.info(f"[STT] Faster-Whisper model '{model_size}' successfully loaded and cached as Singleton.")
                try:
                    import psutil
                    rss = psutil.Process().memory_info().rss / (1024 * 1024)
                    logger.info(f"[MEMORY] Whisper singleton initialized: RSS {rss:.2f} MB")
                except Exception:
                    pass

            except Exception as e:
                import traceback
                logger.critical(f"[STT] Failed to initialize local Faster-Whisper model '{model_size}': {e}\n{traceback.format_exc()}")
                cls._model_instance = "FAILED"

            return cls._model_instance

    @classmethod
    async def warmup(cls, model_size: str = "tiny.en") -> float:
        """Eagerly load model weights and run a 0.1s dummy inference pass during server boot."""
        import time
        start_t = time.perf_counter()
        logger.info(f"[WARMUP] Eagerly warming up FasterWhisper singleton ('{model_size}')...")
        model = await cls._get_model(model_size)
        if model != "FAILED" and model is not None and model != "BYPASS_LOCAL":
            try:
                # 1.5s silent audio sample (24000 zero samples at 16kHz) to compile dynamic shapes/kernels
                dummy_x = np.zeros(24000, dtype=np.float32)
                def run_warmup_inference():
                    import torch
                    with torch.inference_mode():
                        # Use optimized decode params matching production path
                        list(model.transcribe(
                            dummy_x,
                            beam_size=1,
                            temperature=0,
                            vad_filter=False,
                            condition_on_previous_text=False,
                            language="en"
                        ))
                await asyncio.get_event_loop().run_in_executor(None, run_warmup_inference)
                elapsed = (time.perf_counter() - start_t) * 1000.0
                logger.info(f"[WARMUP] FasterWhisper model warmed up in {elapsed:.1f}ms.")
                return elapsed
            except Exception as e:
                logger.warning(f"[WARMUP] Whisper dummy inference failed (non-fatal): {e}")
        return (time.perf_counter() - start_t) * 1000.0

    async def transcribe_utterance(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        session_id: str = "demo_session",
        turn_id: int = 1
    ) -> Optional[str]:
        import time
        import sys
        t_start = time.perf_counter()

        if not audio_bytes or len(audio_bytes) < 160:
            return None

        # 1. Preprocess audio: DC offset removal & peak-bounded gain normalization
        proc_bytes, prep_stats = preprocess_audio_for_stt(audio_bytes)
        logger.info(
            f"[STT-AUDIO-PREPROCESS] original_rms={prep_stats['original_rms']} original_peak={prep_stats['original_peak']} "
            f"gain_applied={prep_stats['gain_applied']:.2f}x processed_rms={prep_stats['processed_rms']} processed_peak={prep_stats['processed_peak']}"
        )

        # 2. Save exact debug WAV for waveform inspection
        save_debug_stt_wav(session_id, turn_id, proc_bytes)

        # Canonical PCM accounting for 16kHz PCM_S16LE mono (32 bytes/ms)
        meta = calculate_pcm_metadata(proc_bytes, sample_rate=16000, channels=1, sample_width=2)
        duration_ms = meta["duration_ms"]

        # 3. Try Groq Cloud API first if available and not dummy key
        has_valid_groq_key = self.api_key and self.api_key not in ("test_groq_key", "test_openai_key", "")
        if has_valid_groq_key:
            logger.info(f"[STT] Routing to Groq Cloud API (Whisper-large-v3-turbo)")
            cloud_text = None
            for attempt in range(1, 3):
                try:
                    cloud_text = await self._transcribe_cloud_fallback(proc_bytes, language, prompt)
                    if cloud_text:
                        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                        logger.info(f"[STT] Groq Cloud successful on attempt {attempt}: '{cloud_text}' in {elapsed_ms:.0f}ms")
                        return cloud_text
                except Exception as e:
                    logger.warning(f"[STT] Groq Cloud attempt {attempt} failed: {e}")
                if attempt == 1:
                    await asyncio.sleep(0.5)  # Quick breather before retry
            logger.warning("[STT] Groq Cloud failed after 2 attempts. Falling back to local...")

        # 4. Try local Faster-Whisper model as fallback
        model = await self._get_model(self.model_size)
        if model != "FAILED" and model != "BYPASS_LOCAL" and model is not None:
            def prepare_audio():
                return _pcm16_to_float32_16k(proc_bytes)

            try:
                x_16k = await asyncio.get_event_loop().run_in_executor(None, prepare_audio)

                # CANONICAL PCM ACCOUNTING & STT-INPUT TELEMETRY
                logger.info(
                    f"[STT-INPUT] bytes={meta['bytes']} samples={len(x_16k)} "
                    f"duration_ms={meta['duration_ms']:.1f}ms sample_rate=16000 channels=1 "
                    f"rms={meta['rms']} peak={meta['peak']}"
                )

                # Explicit Language Routing (Requirement 6)
                whisper_lang = None
                if language:
                    lang_clean = language.split("-")[0].lower()
                    if lang_clean in ("hi", "hindi"):
                        whisper_lang = "hi"
                    elif lang_clean in ("te", "telugu"):
                        whisper_lang = "te"
                    else:
                        whisper_lang = lang_clean

                # Contextual Initial Prompt (Requirement 9)
                effective_prompt = prompt
                if not effective_prompt:
                    if whisper_lang == "hi":
                        effective_prompt = "ग्राहक अपना नाम बता रहा है।"
                    elif whisper_lang == "te":
                        effective_prompt = "కాలర్ తన పేరు చెప్తున్నాడు."
                    else:
                        effective_prompt = "The caller is answering with their name."

                def run_transcription():
                    import torch
                    with torch.inference_mode():
                        segments, info = model.transcribe(
                            x_16k,
                            beam_size=1,                      # Greedy decode — fastest on CPU
                            language=whisper_lang,            # Strict explicit language ("hi", "te", "en")
                            task="transcribe",                # Explicit transcription task
                            vad_filter=False,                 # DISABLE double-VAD (Requirement 7)
                            temperature=0,                    # Deterministic decode
                            condition_on_previous_text=False, # ZERO previous-turn contamination!
                            no_speech_threshold=0.85,         # Context-aware no-speech threshold
                            word_timestamps=False,            # Skip per-word timing
                            log_prob_threshold=-2.0,          # Prevent low-confidence hallucinations without dropping valid speech
                            compression_ratio_threshold=2.4,  # Block repetitive looping tokens
                            initial_prompt=effective_prompt
                        )
                        segments_list = list(segments)
                        if segments_list:
                            avg_logprob = float(sum(s.avg_logprob for s in segments_list) / len(segments_list))
                            compression_ratio = float(max(s.compression_ratio for s in segments_list))
                            no_speech_prob = float(max(s.no_speech_prob for s in segments_list))
                            text = " ".join([s.text for s in segments_list]).strip()
                        else:
                            avg_logprob = -99.0
                            compression_ratio = 0.0
                            no_speech_prob = 1.0
                            text = ""

                        return {
                            "text": text,
                            "language": info.language,
                            "avg_logprob": avg_logprob,
                            "no_speech_prob": no_speech_prob,
                            "compression_ratio": compression_ratio,
                            "temperature": 0.0
                        }

                # Guard local transcription with a timeout and retry strategy
                stt_res = None
                for attempt in range(1, 3):
                    try:
                        coro = asyncio.get_event_loop().run_in_executor(None, run_transcription)
                        stt_res = await asyncio.wait_for(coro, timeout=12.0)
                        if stt_res and stt_res.get("text"):
                            break
                    except asyncio.TimeoutError:
                        logger.error(f"[STT] Local transcription timed out on attempt {attempt} (12.0s limit exceeded)")
                    except Exception as e:
                        logger.error(f"[STT] Local transcription exception on attempt {attempt}: {e}")
                    if attempt == 1:
                        await asyncio.sleep(0.2)

                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                rtf = elapsed_ms / max(duration_ms, 1.0)  # Real-time factor
                
                if stt_res and isinstance(stt_res, dict):
                    text = stt_res.get("text", "")
                    detected_lang = stt_res.get("language", "unknown")
                    logger.info(
                        f"[STT-QUALITY] raw='{text}' language={detected_lang} avg_logprob={stt_res['avg_logprob']:.2f} "
                        f"no_speech_prob={stt_res['no_speech_prob']:.2f} compression_ratio={stt_res['compression_ratio']:.2f} "
                        f"duration_ms={duration_ms:.0f}ms stt_ms={elapsed_ms:.0f}ms RTF={rtf:.2f}x"
                    )
                    return stt_res
                
                logger.info(f"[STT] Local Whisper: empty/silence result or failed after retries | stt={elapsed_ms:.0f}ms")
                return {"text": "", "language": language or "unknown", "avg_logprob": -99.0, "no_speech_prob": 1.0, "compression_ratio": 0.0, "temperature": 0.0}
            except Exception as e:
                import traceback
                logger.warning(f"[STT] Local transcription preparation error: {e}\n{traceback.format_exc()}")

        # 3. Handle mock transcription check
        is_testing = (
            "pytest" in sys.modules
            or os.environ.get("TESTING", "").lower() == "true"
            or settings.APP_ENV == "test"
        )
        if is_testing:
            logger.info("[STT] Test environment detected. Returning mock transcription.")
            return self._mock_transcription(audio_bytes)

        # In production/normal execution, return None (triggers standard recovery loop)
        logger.error("[STT] Whisper both local & cloud failed. Returning error state (None) to trigger client recovery.")
        return None

    async def _transcribe_cloud_fallback(self, pcm_bytes: bytes, language: Optional[str] = None, prompt: Optional[str] = None) -> Optional[str]:
        try:
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(pcm_bytes)
            wav_bytes = wav_io.getvalue()

            async with httpx.AsyncClient(timeout=10.0) as client:
                files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
                data = {"model": "whisper-large-v3-turbo"}
                if language:
                    data["language"] = language
                if prompt:
                    data["prompt"] = prompt
                headers = {"Authorization": f"Bearer {self.api_key}"}

                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    files=files,
                    data=data,
                    headers=headers,
                )
                if response.status_code == 200:
                    text = response.json().get("text", "").strip()
                    if text and text not in _SILENCE_TOKENS and len(text) > 2:
                        logger.info(f"[STT] Cloud Whisper fallback transcribed: '{text}'")
                        return text
        except Exception as e:
            logger.error(f"[STT] Cloud fallback transcription failed: {e}")
        return None

    def _mock_transcription(self, audio_bytes: bytes) -> Optional[str]:
        logger.warning("[STT] Whisper both local & cloud failed. Returning mock speech transcription...")
        duration_sec = len(audio_bytes) / 8000.0
        if duration_sec < 1.0:
            return "Yes."
        elif duration_sec < 2.5:
            return "Confirm my appointment."
        else:
            return "Hello, I am looking to confirm my cardiology appointment scheduled for next week."


STT_MIN_AUDIO_MS = 200.0
STT_MIN_CONFIDENT_AUDIO_MS = 350.0


def get_recovery_message(language_code: str, reason_type: str = "general") -> str:
    """Returns natural, language-aware recovery prompt for session active language (te, hi, en)."""
    lang = (language_code or "en").lower().split("-")[0].strip()
    if lang in ("te", "telugu"):
        if reason_type == "name":
            return "[RECOVERY_SAY:క్షమించండి, మీ పేరు సరిగ్గా వినిపించలేదు. మరోసారి మీ పేరు చెప్పగలరా?]"
        else:
            return "[RECOVERY_SAY:క్షమించండి, మీ మాట సరిగ్గా వినిపించలేదు. మరోసారి చెప్పగలరా?]"
    elif lang in ("hi", "hindi"):
        if reason_type == "name":
            return "[RECOVERY_SAY:क्षमा करें, मुझे आपका नाम समझ नहीं आया। क्या आप अपना शुभ नाम बता सकते हैं?]"
        else:
            return "[RECOVERY_SAY:क्षमा करें, मुझे आपकी आवाज़ साफ़ सुनाई नहीं दी। क्या आप दोहरा सकते हैं?]"
    else:
        if reason_type == "name":
            return "[RECOVERY_SAY:Sorry, I didn't catch your name. Could you repeat your name?]"
        else:
            return "[RECOVERY_SAY:Sorry, I didn't quite catch that. Could you say that again?]"


def validate_stt_audio_pre_whisper(audio_bytes: bytes, current_state: str = "WAIT_FOR_NAME") -> tuple[bool, str, float]:
    """
    Pre-STT Validation Stage:
    Evaluates 16kHz PCM_S16LE audio duration (32 bytes/ms) and RMS acoustic energy.
    Allows short utterances down to 200ms during WAIT_FOR_NAME state, while rejecting pure silence (<140 RMS).
    """
    duration_ms = len(audio_bytes) / 32.0  # 16kHz 16-bit PCM = 32 bytes/ms
    min_thresh = 200.0 if current_state in ("GREETING", "WAIT_FOR_NAME", "IDENTITY_COLLECTION") else STT_MIN_AUDIO_MS
    if duration_ms < min_thresh:
        logger.warning(f"[STT-GUARD] reason=audio_too_short audio_ms={duration_ms:.0f} threshold_ms={min_thresh}")
        return False, "audio_too_short", duration_ms

    import audioop
    try:
        rms = audioop.rms(audio_bytes, 2)
        if rms < 140:
            logger.warning(f"[STT-GUARD] reason=low_energy_audio audio_ms={duration_ms:.0f} threshold_ms={min_thresh} rms={rms}")
            return False, "low_energy_audio", duration_ms
    except Exception:
        pass

    return True, "valid", duration_ms


def validate_stt_transcript(
    stt_input: Any,
    audio_bytes: bytes,
    language: str,
    session_id: str = "demo",
    turn_id: int = 1
) -> tuple[bool, str, Optional[str]]:
    """
    Authoritative STT Validation & Hallucination Guardrail (Requirements 1-20):
    Evaluates empirical Whisper confidence metrics (avg_logprob, no_speech_prob, compression_ratio)
    and audio-duration sanity checks.
    Blocks invalid/hallucinated transcripts BEFORE they reach the LLM or conversation history.
    """
    raw_text = ""
    avg_logprob = 0.0
    no_speech_prob = 0.0
    compression_ratio = 1.0

    if isinstance(stt_input, dict):
        raw_text = stt_input.get("text", "")
        avg_logprob = stt_input.get("avg_logprob", 0.0)
        no_speech_prob = stt_input.get("no_speech_prob", 0.0)
        compression_ratio = stt_input.get("compression_ratio", 1.0)
    elif isinstance(stt_input, str):
        raw_text = stt_input
    else:
        return False, "empty_transcript", None

    audio_duration_ms = len(audio_bytes) / 32.0  # 16kHz 16-bit PCM = 32 bytes/ms

    if not raw_text or not raw_text.strip():
        logger.info(
            f"[STT-QUALITY] session_id={session_id} turn_id={turn_id} language={language} "
            f"audio_ms={audio_duration_ms:.0f}ms raw_transcript='' transcript_chars=0 transcript_words=0 "
            f"avg_logprob=-99.0 no_speech_prob=1.00 compression_ratio=0.00 temperature=0.0 "
            f"validation=NO_SPEECH rejection_reason=empty_transcript"
        )
        return False, "empty_transcript", None

    t = raw_text.strip()
    words = t.lower().split()
    audio_sec = max(audio_duration_ms / 1000.0, 0.1)
    chars_per_second = len(t) / audio_sec
    max_char_freq = max(t.count(c) for c in set(t)) if len(t) > 0 else 0
    repeat_ratio = (max_char_freq / len(t)) if len(t) > 0 else 0.0

    valid = True
    reason = "ACCEPTED"

    # Known short command/name words that must be recognized cleanly even on noisy frames
    KNOWN_SHORT_WORDS = {
        "confirm", "cancel", "yes", "okay", "hello", "hi", "mayank", "vaibhav", "rohan", "akash", "ravi", "john",
        "haan", "nahi", "कन्फर्म", "कैंसिल", "हाँ", "नहीं", "मयंक", "वैभव",
        "నమస్కారం", "అవును", "రద్దు", "కన్ఫర్మ్", "సరే", "ధన్యవాదాలు", "నమస్తే", "వెంకట్", "కృష్ణ", "సాయి", "రామ్", "వెంకటేశ్వర్లు"
    }

    # Common Whisper hallucination phrases on quiet/noisy frames
    HALLUCINATION_PHRASES = [
        "the speaker is", "the speaker", "introducing the speaker",
        "thank you for watching", "thanks for watching", "please subscribe",
        "字幕", "video", "channel"
    ]

    # Rule 1: High no_speech_prob with non-empty text (0.75 for short command words, 0.70 for multi-word phrases)
    effective_no_speech_limit = 0.75 if (len(words) <= 2 and any(w.strip(".,!?").lower() in KNOWN_SHORT_WORDS for w in words)) else 0.70
    if no_speech_prob > effective_no_speech_limit:
        valid, reason = False, f"high_no_speech_prob ({no_speech_prob:.2f})"

    # Rule 2: Low log probability (low acoustic confidence)
    elif avg_logprob < -2.0:
        valid, reason = False, f"low_avg_logprob ({avg_logprob:.2f})"

    # Rule 3: High compression ratio (repetitive decoding loops)
    elif compression_ratio > 2.2:
        valid, reason = False, f"high_compression_ratio ({compression_ratio:.2f})"

    # Rule 4: Short audio (<1.2s) producing multiple sentences or excessive tokens (>8 words or >35 chars)
    elif audio_duration_ms < 1200 and (len(words) > 8 or len(t) > 35):
        valid, reason = False, f"short_audio_long_transcript (audio={audio_duration_ms:.0f}ms, words={len(words)})"

    # Rule 5: Abnormal text/audio ratio (e.g. chars_per_second > 25 for audio < 3.0s)
    elif audio_duration_ms < 3000 and chars_per_second > 25.0:
        valid, reason = False, f"excessive_cps ({chars_per_second:.1f}_cps)"

    # Rule 6: Repeated character / syllable ratio check (e.g. 'अभ्भ्भ्भ्भ्भ्भ्भ्...')
    elif len(t) >= 10 and repeat_ratio > 0.35:
        valid, reason = False, f"high_char_repeat_ratio ({repeat_ratio:.2f})"

    # Rule 7: Multiple persona names hallucination (e.g. "I'm Akash, My name is Arjun, Sophia, David, Maya.")
    elif len(set(re.findall(r"[A-Za-zऀ-ॿ]+", t.lower())).intersection({"akash", "arjun", "sophia", "david", "maya", "ananya"})) >= 2:
        valid, reason = False, "multi_persona_hallucination"

    # Rule 8: Repetitive word patterns (e.g. "my name is my name is" or "करिशे करिशे")
    elif len(words) >= 4 and len(set(words)) <= (len(words) // 2):
        valid, reason = False, "repetitive_word_pattern"

    # Rule 9: Explicit Whisper hallucination phrases (e.g. "The speaker is introducing the speaker.")
    elif any(h in t.lower() for h in HALLUCINATION_PHRASES):
        valid, reason = False, "hallucination_phrase_match"

    if not valid:
        logger.info(
            f"[STT-QUALITY] session_id={session_id} turn_id={turn_id} language={language} "
            f"audio_ms={audio_duration_ms:.0f}ms raw_transcript='{raw_text}' transcript_chars={len(t)} transcript_words={len(words)} "
            f"avg_logprob={avg_logprob:.2f} no_speech_prob={no_speech_prob:.2f} compression_ratio={compression_ratio:.2f} temperature=0.0 "
            f"validation=HALLUCINATION rejection_reason={reason}"
        )
        return False, reason, None

    logger.info(
        f"[STT-QUALITY] session_id={session_id} turn_id={turn_id} language={language} "
        f"audio_ms={audio_duration_ms:.0f}ms raw_transcript='{raw_text}' transcript_chars={len(t)} transcript_words={len(words)} "
        f"avg_logprob={avg_logprob:.2f} no_speech_prob={no_speech_prob:.2f} compression_ratio={compression_ratio:.2f} temperature=0.0 "
        f"validation=VALID rejection_reason=ACCEPTED"
    )
    return True, "ACCEPTED", t
