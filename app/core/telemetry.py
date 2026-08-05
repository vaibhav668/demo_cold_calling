from typing import Dict, Any

# Global telemetry metrics dictionary for startup boot times
STARTUP_METRICS: Dict[str, Any] = {
    "boot_time_sec": 0.0,
    "vad_load_ms": 0.0,
    "stt_load_ms": 0.0,
    "tts_load_ms": 0.0,
    "llm_warmup_ms": 0.0,
    "total_warmup_ms": 0.0,
    "rss_mb": 0.0
}
