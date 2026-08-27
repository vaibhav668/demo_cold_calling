# Enable UTF-8 encoding by default on Windows to prevent UnicodeDecodeError in external libraries (like kokoro-onnx)
import builtins
import sys

if sys.platform == "win32":
    _original_open = builtins.open
    def _utf8_open(file, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if len(args) > 0 else "r")
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return _original_open(file, *args, **kwargs)
    builtins.open = _utf8_open

# Global NumPy 2.x compatibility patch for Kokoro ONNX voice pickle loading
import numpy as np
_original_np_load = np.load
def _patched_np_load(*args, **kwargs):
    if "allow_pickle" not in kwargs:
        kwargs["allow_pickle"] = True
    return _original_np_load(*args, **kwargs)
np.load = _patched_np_load

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "AI Voice Calling Platform"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Databases
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_cold_call"
    VECTOR_DB_PROVIDER: str = "chroma"
    CHROMA_DB_PATH: str = "./data/chroma"

    # Telephony
    TELEPHONY_PROVIDER: str = "plivo"
    PLIVO_AUTH_ID: Optional[str] = None
    PLIVO_AUTH_TOKEN: Optional[str] = None
    PLIVO_PHONE_NUMBER: Optional[str] = None

    # AI
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    LLM_PROVIDER: str = "gemini"
    FALLBACK_LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "gemini-3.5-flash"

    # Speech AI providers
    STT_PROVIDER: str = "faster_whisper"
    WHISPER_MODEL: str = "large-v3-turbo"
    VAD_PROVIDER: str = "silero"
    TTS_PROVIDER: str = "svara"
    EMBEDDING_PROVIDER: str = "bge_m3"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # Svara TTS specifics
    SVARA_MODEL_PATH: str = "./models/svara/svara-tts-v1.Q4_K_M.gguf"
    SVARA_THREADS: int = 4
    SVARA_CONTEXT_SIZE: int = 2048
    SVARA_DEFAULT_SPEED: float = 1.0
    
    # Svara Female Voice Mapping (centralized female-only mapping)
    SVARA_VOICE_SOPHIA_EN: str = "svara_en_female_sophia"
    SVARA_VOICE_SOPHIA_HI: str = "svara_hi_female_sophia"
    SVARA_VOICE_SOPHIA_TE: str = "svara_te_female_sophia"

    SVARA_VOICE_MAYA_EN: str = "svara_en_female_maya"
    SVARA_VOICE_MAYA_HI: str = "svara_hi_female_maya"
    SVARA_VOICE_MAYA_TE: str = "svara_te_female_maya"

    SVARA_VOICE_EMMA_EN: str = "svara_en_female_emma"
    SVARA_VOICE_EMMA_HI: str = "svara_hi_female_emma"
    SVARA_VOICE_EMMA_TE: str = "svara_te_female_emma"

    SVARA_VOICE_ANANYA_EN: str = "svara_en_female_ananya"
    SVARA_VOICE_ANANYA_HI: str = "svara_hi_female_ananya"
    SVARA_VOICE_ANANYA_TE: str = "svara_te_female_ananya"

    # Legacy Kokoro settings retained for fallback compatibility
    KOKORO_MODEL_DIR: str = "./models/kokoro"
    KOKORO_DEFAULT_SPEED: float = 1.05
    KOKORO_VOICE_SOPHIA_EN: str = "af_nicole"
    KOKORO_VOICE_SOPHIA_HI: str = "hf_alpha"
    KOKORO_VOICE_SOPHIA_TE: str = "af_nicole"
    KOKORO_VOICE_MAYA_EN: str = "af_sky"
    KOKORO_VOICE_MAYA_HI: str = "hf_beta"
    KOKORO_VOICE_MAYA_TE: str = "af_sky"
    KOKORO_VOICE_ANANYA_EN: str = "af_bella"
    KOKORO_VOICE_ANANYA_HI: str = "hf_alpha"
    KOKORO_VOICE_ANANYA_TE: str = "af_bella"
    KOKORO_VOICE_ARJUN_EN: str = "am_michael"
    KOKORO_VOICE_ARJUN_HI: str = "hm_omega"
    KOKORO_VOICE_ARJUN_TE: str = "am_michael"
    KOKORO_VOICE_DAVID_EN: str = "am_adam"
    KOKORO_VOICE_DAVID_HI: str = "hm_psi"
    KOKORO_VOICE_DAVID_TE: str = "am_adam"


    # Security & Authentication
    JWT_SECRET_KEY: str = "supersecretdevelopmentkeychangeinproduction"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

    @classmethod
    def validate_critical_settings(cls, values: dict) -> dict:
        db_url = values.get("DATABASE_URL")
        if not db_url or db_url.strip() == "":
            raise ValueError("DATABASE_URL environment variable is missing or empty. Application startup aborted.")
        return values

settings = Settings()
# Custom manual check to trigger clear error message on init
if not settings.DATABASE_URL or settings.DATABASE_URL.strip() == "":
    raise ValueError("DATABASE_URL environment variable is missing or empty. Application startup aborted.")


def check_low_memory() -> bool:
    """
    Unified check to determine if the system is running in a memory-constrained
    environment (e.g. Render 512MB container, cgroups limit, or low host RAM).
    """
    import os
    # 1. Explicit environment overrides or platform indicators
    if os.environ.get("LOW_MEMORY_DEPLOYMENT", "false").lower() == "true":
        return True
    if os.environ.get("RENDER") or os.environ.get("RENDER_SERVICE_ID") or os.environ.get("RENDER_INSTANCE_ID"):
        return True

    # 2. Check cgroups memory limit (accurate for container limits like Render)
    for path in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory.high"]:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    val = f.read().strip()
                    if val and val.isdigit():
                        limit = int(val)
                        if limit < 1024 * 1024 * 1024:  # < 1GB limit
                            return True
        except Exception:
            pass

    # 3. Check psutil as fallback
    try:
        import psutil
        if psutil.virtual_memory().total < 1024 * 1024 * 1024:
            return True
    except Exception:
        pass

    return False

