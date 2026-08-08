import os
import sys

# Configure environment variables to use persistent /app/models/ cache paths
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
    print("Pre-downloading faster-whisper base model (fallback)...", flush=True)
    from faster_whisper import WhisperModel
    WhisperModel("base", device="cpu", compute_type="int8")
    print("Pre-downloading faster-whisper large-v3-turbo model...", flush=True)
    WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    print("Whisper models successfully cached.", flush=True)
except Exception as e:
    print(f"Error caching Whisper models: {e}", file=sys.stderr, flush=True)

# 2. Preload SentenceTransformer Embedding models
try:
    print("Pre-downloading sentence-transformers all-MiniLM-L6-v2 model...", flush=True)
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print("Pre-downloading BAAI/bge-m3 model...", flush=True)
    SentenceTransformer("BAAI/bge-m3", device="cpu")
    print("Embedding models successfully cached.", flush=True)
except Exception as e:
    print(f"Error caching embedding models: {e}", file=sys.stderr, flush=True)

# 3. Preload Kokoro TTS ONNX model and voices
try:
    print("Pre-downloading Kokoro ONNX model files...", flush=True)
    import urllib.request
    
    # We'll download to the default target directory
    target_dir = os.environ.get("KOKORO_MODEL_DIR", "/app/models/kokoro" if os.path.exists("/app") else "./models/kokoro")
    os.makedirs(target_dir, exist_ok=True)
    
    onnx_path = os.path.join(target_dir, "kokoro-v1.0.onnx")
    voices_path = os.path.join(target_dir, "voices.json")
    
    onnx_valid = os.path.exists(onnx_path) and os.path.getsize(onnx_path) >= 250 * 1024 * 1024
    if not onnx_valid:
        if os.path.exists(onnx_path):
            print(f"Incomplete/truncated model file found (size: {os.path.getsize(onnx_path)} bytes). Deleting and re-downloading...", flush=True)
            try:
                os.remove(onnx_path)
            except Exception:
                pass
        print(f"Downloading Kokoro ONNX model to {onnx_path}...", flush=True)
        url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        urllib.request.urlretrieve(url, onnx_path)
    
    voices_valid = os.path.exists(voices_path) and os.path.getsize(voices_path) >= 20 * 1024 * 1024
    if not voices_valid:
        if os.path.exists(voices_path):
            print("Incomplete/truncated voices file found. Deleting and re-downloading...", flush=True)
            try:
                os.remove(voices_path)
            except Exception:
                pass
        print(f"Downloading Kokoro Voices database to {voices_path}...", flush=True)
        url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"
        urllib.request.urlretrieve(url, voices_path)
        
    print("Kokoro model files successfully cached.", flush=True)
except Exception as e:
    print(f"Error caching Kokoro TTS models: {e}", file=sys.stderr, flush=True)

print("Preloading complete!", flush=True)
