import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "AI Voice Demo Platform"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Databases (purely in-memory RAG)
    VECTOR_DB_PROVIDER: str = "chroma"
    CHROMA_DB_PATH: str = ":memory:"

    # AI
    GROQ_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    LLM_PROVIDER: str = "groq"
    FALLBACK_LLM_PROVIDER: str = "openrouter"
    LLM_MODEL: str = "llama-3.1-8b-instant"

    # Speech AI providers
    STT_PROVIDER: str = "faster_whisper"
    WHISPER_MODEL: str = "tiny"  # tiny supports English, Hindi, Telugu natively
    VAD_PROVIDER: str = "silero"
    TTS_PROVIDER: str = "edge_tts"
    EMBEDDING_PROVIDER: str = "bge_m3"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

settings = Settings()

def check_low_memory() -> bool:
    """
    Check if the system is running in a memory-constrained
    environment (e.g. Render 512MB container, cgroups limit, or low host RAM).
    """
    # 1. Explicit environment overrides or platform indicators
    if os.environ.get("LOW_MEMORY_DEPLOYMENT", "true").lower() == "true":
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
