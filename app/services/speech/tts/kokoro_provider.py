import os
import io
import asyncio
import urllib.request
import audioop
import numpy as np
from typing import AsyncGenerator, Optional
from concurrent.futures import ThreadPoolExecutor
from app.core.logging import logger
from app.core.config import settings
from app.services.speech.tts.base import TextToSpeechProvider

_kokoro_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kokoro_tts")

# Simple Telugu mapping to English phonetics (for reading Telugu in English voice)
_TELUGU_MAP = {
    'అ': 'a', 'ఆ': 'aa', 'ఇ': 'i', 'ఈ': 'ee', 'ఉ': 'u', 'ఊ': 'oo', 'ఋ': 'ri',
    'ఎ': 'e', 'ఏ': 'ee', 'ఐ': 'ai', 'ఒ': 'o', 'ఓ': 'oo', 'ఔ': 'au', 'అం': 'an', 'అః': 'ah',
    'ా': 'aa', 'ి': 'i', 'ీ': 'ee', 'ు': 'u', 'ూ': 'oo', 'ృ': 'ri',
    'ె': 'e', 'ే': 'ee', 'ై': 'ai', 'ொ': 'o', 'ో': 'oo', 'ౌ': 'au', 'ం': 'n', 'ః': 'h',
    '్': '', # Telugu halant/virama maps to empty string to drop inherent vowel
    'క': 'ka', 'ఖ': 'kha', 'గ': 'ga', 'ఘ': 'gha', 'ఙ': 'nga',
    'చ': 'cha', 'ఛ': 'chha', 'జ': 'ja', 'ఝ': 'jha', 'ఞ': 'nya',
    'ట': 'ta', 'ఠ': 'tha', 'డ': 'da', 'ఢ': 'dha', 'ణ': 'na',
    'త': 'ta', 'థ': 'tha', 'ద': 'da', 'ధ': 'dha', 'న': 'na',
    'ప': 'pa', 'ఫ': 'pha', 'బ': 'ba', 'భ': 'bha', 'మ': 'ma',
    'య': 'ya', 'ర': 'ra', 'ల': 'la', 'వ': 'va', 'శ': 'sha',
    'ष': 'sha', 'स': 'sa', 'ह': 'ha', 'ళ': 'la', 'క్ష': 'ksha',
    '౦': '0', '౧': '1', '౨': '2', '౩': '3', '౪': '4', '౫': '5', '౬': '6', '౭': '7', '౮': '8', '౯': '9',
    ' ': ' '
}

def transliterate_telugu(text: str) -> str:
    """Detects Telugu script and transliterates to phonetic Latin script."""
    has_telugu = any('\u0c00' <= char <= '\u0c7F' for char in text)
    if not has_telugu:
        return text

    result = []
    for char in text:
        result.append(_TELUGU_MAP.get(char, char))

    transliterated = "".join(result)
    transliterated = transliterated.replace("aae", "e").replace("aai", "ai").replace("aao", "o")
    transliterated = transliterated.replace("aaa", "aa")
    return transliterated


# Simple Hindi mapping to English phonetics (for reading Hindi in English voice)
_HINDI_MAP = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'an', 'अः': 'ah',
    'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', 'ः': 'h',
    '्': '', # Devanagari halant/virama maps to empty string to drop inherent vowel
    '।': '.', # Devanagari danda maps to period
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'sha',
    'ष': 'sha', 'स': 'sa', 'ह': 'ha', 'ळ': 'la', 'क्ष': 'ksha',
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    ' ': ' '
}

def transliterate_hindi(text: str) -> str:
    """Detects Devanagari script and transliterates to phonetic Latin script."""
    has_hindi = any('\u0900' <= char <= '\u097F' for char in text)
    if not has_hindi:
        return text

    result = []
    for char in text:
        result.append(_HINDI_MAP.get(char, char))

    transliterated = "".join(result)
    transliterated = transliterated.replace("aae", "e").replace("aai", "ai").replace("aao", "o")
    transliterated = transliterated.replace("aaa", "aa")
    return transliterated


class KokoroProvider(TextToSpeechProvider):
    """
    Local Text-to-Speech provider utilizing Kokoro-82M ONNX.
    Optimized for multi-language (EN, HI, TE) real-time streaming.
    """

    _model_instance = None
    _model_lock = asyncio.Lock()
    _files_verified: bool = False  # Skip repeated file-check I/O after first verification
    _style_cache: dict = {}

    @classmethod
    async def warmup(cls) -> float:
        """Eagerly verify files and initialize ONNX session on boot."""
        import time
        start_t = time.perf_counter()
        logger.info("[WARMUP] Eagerly warming up KokoroProvider...")
        instance = cls()
        if not cls._files_verified:
            await asyncio.get_event_loop().run_in_executor(None, instance._download_files)
            cls._files_verified = True
        model = await cls._get_model(instance.model_dir, instance.onnx_path, instance.voices_path)

        if model and model != "FAILED":
            from app.core.config import settings as _settings
            # All production voices that could be requested in the demo
            PRODUCTION_VOICES = [
                _settings.KOKORO_VOICE_SOPHIA_EN,
                _settings.KOKORO_VOICE_MAYA_EN,
                _settings.KOKORO_VOICE_ANANYA_EN,
                _settings.KOKORO_VOICE_ARJUN_EN,
                _settings.KOKORO_VOICE_DAVID_EN,
            ]
            # Filter to available voices only
            available = set()
            try:
                if hasattr(model, "voices") and isinstance(model.voices, dict):
                    available = set(model.voices.keys())
                elif hasattr(model, "get_voices"):
                    available = set(model.get_voices())
            except Exception:
                pass

            logger.info(f"[WARMUP] Pre-caching style arrays for {len(PRODUCTION_VOICES)} production voices...")
            for voice_id in PRODUCTION_VOICES:
                if voice_id not in cls._style_cache:
                    try:
                        if not available or voice_id in available:
                            style = model.get_voice_style(voice_id)
                            cls._style_cache[voice_id] = style
                            logger.info(f"[WARMUP] Cached voice style: '{voice_id}'")
                    except Exception as ve:
                        logger.warning(f"[WARMUP] Could not cache style for '{voice_id}': {ve}")

            # Run dummy synthesis using the actual default production voice (af_nicole = Sophia EN)
            primary_voice = cls._style_cache.get(
                _settings.KOKORO_VOICE_SOPHIA_EN,
                _settings.KOKORO_VOICE_SOPHIA_EN
            )
            try:
                logger.info("[WARMUP] Running dummy Kokoro synthesis on primary production voice...")
                def _dummy_run():
                    return model.create("Hello, warming up.", voice=primary_voice, speed=1.0, lang="en-us")
                await asyncio.get_event_loop().run_in_executor(_kokoro_executor, _dummy_run)
                logger.info("[WARMUP] Dummy Kokoro synthesis completed.")
            except Exception as e:
                logger.warning(f"[WARMUP] Dummy Kokoro synthesis failed: {e}")

        elapsed = (time.perf_counter() - start_t) * 1000.0
        logger.info(f"[WARMUP] KokoroProvider warmed up in {elapsed:.1f}ms.")
        return elapsed

    def __init__(self) -> None:
        self.model_dir = settings.KOKORO_MODEL_DIR
        self.onnx_path = os.path.join(self.model_dir, "kokoro-v1.0.onnx")
        self.voices_path = os.path.join(self.model_dir, "voices.json")

    def _download_files(self) -> None:
        """Download Kokoro ONNX model and voice packs if missing or truncated."""
        import json
        os.makedirs(self.model_dir, exist_ok=True)
        
        # kokoro-v1.0.onnx is ~320MB. Check for minimum 250MB to detect truncated files.
        onnx_valid = os.path.exists(self.onnx_path) and os.path.getsize(self.onnx_path) >= 250 * 1024 * 1024
        if not onnx_valid:
            if os.path.exists(self.onnx_path):
                logger.warning(f"[Kokoro] Incomplete/truncated model file found (size: {os.path.getsize(self.onnx_path)} bytes). Deleting and re-downloading...")
                try:
                    os.remove(self.onnx_path)
                except Exception:
                    pass
            logger.info(f"[Kokoro] Downloading model to {self.onnx_path}...")
            url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
            urllib.request.urlretrieve(url, self.onnx_path)
            logger.info("[Kokoro] Model downloaded successfully.")

        # voices.json is ~30MB. Must be valid UTF-8 JSON.
        voices_valid = False
        if os.path.exists(self.voices_path) and os.path.getsize(self.voices_path) >= 20 * 1024 * 1024:
            try:
                with open(self.voices_path, "r", encoding="utf-8") as f:
                    json.load(f)
                voices_valid = True
            except Exception as e:
                logger.warning(f"[Kokoro] Corrupted voices.json detected: {e}. Re-downloading...")

        if not voices_valid:
            if os.path.exists(self.voices_path):
                try:
                    os.remove(self.voices_path)
                except Exception:
                    pass
            logger.info(f"[Kokoro] Downloading JSON voice database to {self.voices_path}...")
            url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"
            urllib.request.urlretrieve(url, self.voices_path)
            logger.info("[Kokoro] Voices database downloaded successfully.")

    @classmethod
    async def _get_model(cls, model_dir: str, onnx_path: str, voices_path: str):
        """Loads and caches the Kokoro model as a thread-safe singleton."""
        if cls._model_instance is not None:
            return cls._model_instance

        async with cls._model_lock:
            if cls._model_instance is not None:
                return cls._model_instance

            try:
                import json
                import onnxruntime as ort
                from kokoro_onnx import Kokoro

                # 1. Startup Diagnostics & Asset Audit
                onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024) if os.path.exists(onnx_path) else 0
                bin_path = os.path.join(model_dir, "voices.bin")
                json_path = os.path.join(model_dir, "voices.json")

                bin_size_mb = os.path.getsize(bin_path) / (1024 * 1024) if os.path.exists(bin_path) else 0
                json_size_mb = os.path.getsize(json_path) / (1024 * 1024) if os.path.exists(json_path) else 0

                logger.info(f"[Kokoro-Audit] Resolved model path:      {onnx_path} ({onnx_size_mb:.1f} MB)")
                logger.info(f"[Kokoro-Audit] Resolved binary asset:    {bin_path} ({bin_size_mb:.1f} MB)")
                logger.info(f"[Kokoro-Audit] Resolved JSON asset:      {json_path} ({json_size_mb:.1f} MB)")

                # 2. Strict Pre-flight Validation
                if not os.path.exists(onnx_path) or onnx_size_mb < 250:
                    raise RuntimeError(f"Kokoro initialization failed: ONNX model asset missing or truncated at {onnx_path}")

                if not os.path.exists(bin_path) and not os.path.exists(json_path):
                    raise RuntimeError(f"Kokoro initialization failed: Both voices.bin and voices.json are missing in {model_dir}")

                # 3. Determine available ONNX execution providers (GPU vs CPU)
                available = ort.get_available_providers()
                providers = []
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                providers.append("CPUExecutionProvider")
                logger.info(f"[Kokoro-Audit] ONNX available providers: {available} | Selected: {providers}")

                # CPU thread optimization options
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                # Initialize inference session
                session = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
                )

                # 4. Multi-asset initialization (tries voices.bin first, fallback to voices.json for package version compatibility)
                cls._model_instance = None
                last_err = None

                for target_asset in [bin_path, json_path]:
                    if not os.path.exists(target_asset):
                        continue
                    try:
                        _orig_load = np.load
                        np.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, "allow_pickle": True})
                        instance = Kokoro.from_session(session, target_asset)
                        cls._model_instance = instance
                        logger.info(f"[Kokoro-Audit] ✓ Successfully loaded Kokoro.from_session with asset: {target_asset}")
                        break
                    except Exception as try_err:
                        last_err = try_err
                        logger.debug(f"[Kokoro-Audit] Trial load with {target_asset} failed: {try_err}")
                    finally:
                        np.load = _orig_load

                if cls._model_instance is None:
                    raise RuntimeError(f"Kokoro initialization failed: unable to load Kokoro from session with assets ({last_err})")

                active_ep = session.get_providers()[0]
                voice_list = list(cls._model_instance.get_voices() if hasattr(cls._model_instance, "get_voices") else cls._model_instance.voices.keys())
                logger.info(f"[Kokoro-Audit] ✓ ONNX session active on {active_ep} | Available voices ({len(voice_list)}): {voice_list[:10]}...")

                try:
                    import psutil
                    rss = psutil.Process().memory_info().rss / (1024 * 1024)
                    logger.info(f"[MEMORY] Kokoro singleton initialized: RSS {rss:.2f} MB")
                except Exception:
                    pass

            except Exception as e:
                import traceback
                logger.error(f"[Kokoro] Failed to load local ONNX model: {e}\n{traceback.format_exc()}")
                cls._model_instance = "FAILED"

            return cls._model_instance

    def _resolve_voice_and_lang(self, voice_config: Optional[dict], language: Optional[str]) -> tuple[str, str]:
        """Resolves Kokoro voice preset and G2P language code from request config."""
        lang = (language or "en").split("-")[0].lower()  # 'en', 'hi', 'te'
        
        # Resolve persona name
        persona = "sophia"
        if voice_config:
            persona = (
                voice_config.get("persona_name") or 
                voice_config.get("voice_name") or 
                voice_config.get("name") or "sophia"
            ).lower().strip()
            
        # Determine voice preset mapping
        voice_map = {
            ("sophia", "en"): settings.KOKORO_VOICE_SOPHIA_EN,
            ("sophia", "hi"): settings.KOKORO_VOICE_SOPHIA_HI,
            ("sophia", "te"): settings.KOKORO_VOICE_SOPHIA_TE,
            ("maya", "en"): settings.KOKORO_VOICE_MAYA_EN,
            ("maya", "hi"): settings.KOKORO_VOICE_MAYA_HI,
            ("maya", "te"): settings.KOKORO_VOICE_MAYA_TE,
            ("ananya", "en"): settings.KOKORO_VOICE_ANANYA_EN,
            ("ananya", "hi"): settings.KOKORO_VOICE_ANANYA_HI,
            ("ananya", "te"): settings.KOKORO_VOICE_ANANYA_TE,
            ("arjun", "en"): settings.KOKORO_VOICE_ARJUN_EN,
            ("arjun", "hi"): settings.KOKORO_VOICE_ARJUN_HI,
            ("arjun", "te"): settings.KOKORO_VOICE_ARJUN_TE,
            ("david", "en"): settings.KOKORO_VOICE_DAVID_EN,
            ("david", "hi"): settings.KOKORO_VOICE_DAVID_HI,
            ("david", "te"): settings.KOKORO_VOICE_DAVID_TE,
        }
        
        voice = voice_map.get((persona, lang), settings.KOKORO_VOICE_SOPHIA_EN)
        
        # Determine language code for phonemization — must be supported by kokoro-onnx config
        lang_code = "en-us"  
            
        return voice, lang_code

    async def stream_speech(
        self,
        text: str,
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None,
    ) -> AsyncGenerator[bytes, None]:
        import time
        t_start = time.perf_counter()

        # Handle Telugu script phonetic transliteration
        processed_text = transliterate_telugu(text)
        # Handle Hindi script phonetic transliteration
        processed_text = transliterate_hindi(processed_text)
        
        # Strip any remaining non-ASCII characters to prevent espeak language-switch hangs
        processed_text = "".join(c for c in processed_text if ord(c) < 128)

        # Skip file I/O check if already verified at warmup or previous call
        if not self.__class__._files_verified:
            await asyncio.get_event_loop().run_in_executor(None, self._download_files)
            self.__class__._files_verified = True

        model = await self._get_model(self.model_dir, self.onnx_path, self.voices_path)
        if model == "FAILED" or model is None:
            raise RuntimeError("Local Kokoro TTS loading failed.")

        voice, lang_code = self._resolve_voice_and_lang(voice_config, language)

        # Validate voice exists in Kokoro voices catalog to prevent runtime errors
        if hasattr(model, "voices") and isinstance(model.voices, dict):
            if voice not in model.voices:
                fallback_voice = "af_sky" if "af_sky" in model.voices else (list(model.voices.keys())[0] if model.voices else "af_bella")
                logger.warning(f"[Kokoro] Requested voice '{voice}' not found in loaded catalog. Safely mapping to fallback voice '{fallback_voice}'.")
                voice = fallback_voice

        # Cache voice style numpy array to avoid repeated file system / JSON parsing hits
        if voice not in self.__class__._style_cache:
            try:
                style_array = model.get_voice_style(voice)
                self.__class__._style_cache[voice] = style_array
            except Exception as style_err:
                logger.warning(f"[Kokoro] Failed to cache style for '{voice}': {style_err}. Falling back to name string.")
                self.__class__._style_cache[voice] = voice

        voice_style = self.__class__._style_cache[voice]

        speed = settings.KOKORO_DEFAULT_SPEED
        if voice_config and "speed" in voice_config:
            try:
                speed = float(voice_config["speed"])
            except ValueError:
                pass

        logger.info(f"[TTS-START] Kokoro synthesizing: voice='{voice}' lang='{lang_code}' speed={speed} chars={len(processed_text)}")

        try:
            loop = asyncio.get_event_loop()

            # Pass the cached numpy style array directly to create() to bypass lookup
            def _synthesize_sync():
                return model.create(
                    processed_text,
                    voice=voice_style,
                    speed=speed,
                    lang=lang_code
                )

            samples, sample_rate = await loop.run_in_executor(_kokoro_executor, _synthesize_sync)

            if cancel_event and cancel_event.is_set():
                logger.info("[TTS] Stream cancelled after synthesis.")
                return

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.info(f"[TTS-FIRST-CHUNK] {elapsed_ms:.0f}ms | chars={len(processed_text)}")

            # Convert float32 → int16
            int16_samples = (samples * 32767.0).clip(-32768, 32767).astype(np.int16)
            pcm_24k_bytes = int16_samples.tobytes()

            # Resample 24kHz → 8kHz
            pcm_8k_bytes, _ = audioop.ratecv(pcm_24k_bytes, 2, 1, 24000, 8000, None)

            # Transcode to G.711 mu-law
            mulaw_bytes = bytearray(audioop.lin2ulaw(pcm_8k_bytes, 2))

            # Stream in 480-byte (60ms) frames
            offset = 0
            while offset < len(mulaw_bytes):
                if cancel_event and cancel_event.is_set():
                    return
                chunk = bytes(mulaw_bytes[offset:offset + 480])
                if len(chunk) < 480:
                    chunk = chunk.ljust(480, b'\xff')
                yield chunk
                offset += 480

        except Exception as e:
            logger.error(f"[Kokoro] Generation stream failed: {e}", exc_info=True)
            raise e
