import os
import sys

# Configure environment variables to use persistent cache paths
os.environ["HF_HOME"] = "/app/models/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/models/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/models/hf_cache"
os.environ["XDG_CACHE_HOME"] = "/app/models/xdg_cache"
os.environ["TORCH_HOME"] = "/app/models/torch_cache"

# Monkey-patch missing transformers symbol if sentence-transformers requires it
try:
    import transformers
    if not hasattr(transformers, "is_torch_npu_available"):
        setattr(transformers, "is_torch_npu_available", lambda: False)
except Exception:
    pass

print("Pre-downloading models to bake them into the Docker image...", flush=True)

# 1. Preload Whisper STT models
try:
    print("Pre-downloading faster-whisper tiny model...", flush=True)
    from faster_whisper import WhisperModel
    WhisperModel("tiny", device="cpu", compute_type="int8")
    print("Pre-downloading faster-whisper base model...", flush=True)
    WhisperModel("base", device="cpu", compute_type="int8")
    print("Whisper models successfully cached.", flush=True)
except Exception as e:
    print(f"Error caching Whisper models: {e}", file=sys.stderr, flush=True)

# 2. Preload SentenceTransformer Embedding models
try:
    print("Pre-downloading sentence-transformers all-MiniLM-L6-v2 model...", flush=True)
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print("Embedding models successfully cached.", flush=True)
except Exception as e:
    print(f"Error caching embedding models: {e}", file=sys.stderr, flush=True)

print("Preloading complete!", flush=True)
