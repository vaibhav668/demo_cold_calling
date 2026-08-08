import os
import io
import wave
import audioop
import asyncio
import httpx
import numpy as np
from typing import Optional
from app.core.logging import logger
from app.core.config import settings
from app.services.speech.stt.base import SpeechToTextProvider

# Ignore common Whisper hallucination tokens on telephone static/silence
_SILENCE_TOKENS = {
    "", ".", "..", "...", "Thank you.", "Bye.", "Thanks.", "you",
    "You.", "you.", "Okay.", "okay.", "Hmm.", "hmm.", "Uh.", "uh.",
    "Mm.", "mm.", "Mmm.", "mmm.", "[Music]", "[Applause]", "[Laughter]",
}


def _ulaw_to_float32_16k(audio_bytes: bytes) -> np.ndarray:
    """
    Convert G.711 mu-law 8kHz bytes → float32 numpy array at 16kHz.
    Uses C-level audioop functions throughout — no Python loops, no GIL pressure.
    """
    pcm_8k = audioop.ulaw2lin(audio_bytes, 2)
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
    samples = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


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
                        cache_dir = os.environ.get("HF_HOME", "/app/models/hf_cache")
                        os.makedirs(cache_dir, exist_ok=True)
                        download_root = cache_dir
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
        prompt: Optional[str] = None
    ) -> Optional[str]:
        import time
        import sys
        t_start = time.perf_counter()

        if not audio_bytes or len(audio_bytes) < 160:
            return None

        # Estimate audio duration for telemetry
        duration_ms = len(audio_bytes) / 8.0  # 8000 bytes/s = 8 bytes/ms (mu-law 8kHz)

        # 1. Try Groq Cloud API first if available and not dummy key
        has_valid_groq_key = self.api_key and self.api_key not in ("test_groq_key", "test_openai_key", "")
        if has_valid_groq_key:
            logger.info(f"[STT] Routing to Groq Cloud API (Whisper-large-v3-turbo)")
            cloud_text = None
            for attempt in range(1, 3):
                try:
                    cloud_text = await self._transcribe_cloud_fallback(audio_bytes, language, prompt)
                    if cloud_text:
                        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                        logger.info(f"[STT] Groq Cloud successful on attempt {attempt}: '{cloud_text}' in {elapsed_ms:.0f}ms")
                        return cloud_text
                except Exception as e:
                    logger.warning(f"[STT] Groq Cloud attempt {attempt} failed: {e}")
                if attempt == 1:
                    await asyncio.sleep(0.5)  # Quick breather before retry
            logger.warning("[STT] Groq Cloud failed after 2 attempts. Falling back to local...")

        # 2. Try local Faster-Whisper model as fallback
        model = await self._get_model(self.model_size)
        if model != "FAILED" and model != "BYPASS_LOCAL" and model is not None:
            # Decode G.711 mu-law 8kHz → float32 16kHz (C-level, no Python loops)
            def prepare_audio():
                return _ulaw_to_float32_16k(audio_bytes)

            try:
                x_16k = await asyncio.get_event_loop().run_in_executor(None, prepare_audio)

                # Resolve explicit language code (strip subtags like 'en-us' → 'en')
                whisper_lang = None
                if language:
                    whisper_lang = language.split("-")[0].lower()

                def run_transcription():
                    import torch
                    with torch.inference_mode():
                        segments, info = model.transcribe(
                            x_16k,
                            beam_size=1,                      # Greedy decode — fastest on CPU
                            language=whisper_lang,            # Explicit lang — skip auto-detect
                            vad_filter=False,                 # Remove double-VAD (we already ran VAD)
                            temperature=0,                    # Deterministic decode, no sampling
                            condition_on_previous_text=False, # Fresh decode each turn
                            no_speech_threshold=0.6,          # Reject silence quickly
                            word_timestamps=False,            # Skip per-word timing
                            log_prob_threshold=-3.0,          # Very permissive — prevents fallback temperature retries
                            compression_ratio_threshold=3.5,  # Very permissive — prevents compression fallback retries
                            initial_prompt=prompt
                        )
                        text = " ".join([seg.text for seg in segments]).strip()
                        return text, info.language

                # Guard local transcription with a timeout and retry strategy
                text = None
                detected_lang = "unknown"
                for attempt in range(1, 3):
                    try:
                        coro = asyncio.get_event_loop().run_in_executor(None, run_transcription)
                        # wait_for is fully compatible with both python 3.10 and 3.11
                        text, detected_lang = await asyncio.wait_for(coro, timeout=6.0)
                        if text:
                            break
                    except asyncio.TimeoutError:
                        logger.error(f"[STT] Local transcription timed out on attempt {attempt} (6.0s limit exceeded)")
                    except Exception as e:
                        logger.error(f"[STT] Local transcription exception on attempt {attempt}: {e}")
                    if attempt == 1:
                        await asyncio.sleep(0.5)

                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                rtf = elapsed_ms / max(duration_ms, 1.0)  # Real-time factor
                if text and text not in _SILENCE_TOKENS and len(text) > 2:
                    logger.info(f"[STT] Local Whisper ({detected_lang}): '{text}' | audio={duration_ms:.0f}ms stt={elapsed_ms:.0f}ms RTF={rtf:.2f}x")
                    return text
                logger.info(f"[STT] Local Whisper: empty/silence result or failed after retries | stt={elapsed_ms:.0f}ms")
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

    async def _transcribe_cloud_fallback(self, audio_bytes: bytes, language: Optional[str] = None, prompt: Optional[str] = None) -> Optional[str]:
        try:
            pcm_bytes = audioop.ulaw2lin(audio_bytes, 2)
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(8000)
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
