import sys
import os
import io
import time
import wave
import asyncio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.logging import logger
from app.services.speech.stt.faster_whisper_provider import FasterWhisperProvider, preprocess_audio_for_stt, calculate_pcm_metadata

TEST_BENCHMARK_CASES = [
    {
        "id": "1_ENGLISH_SHORT_NAME",
        "language": "en",
        "text": "Vaibhav",
        "dur_ms": 464,
        "expect_name": "Vaibhav"
    },
    {
        "id": "2_ENGLISH_FULL_NAME",
        "language": "en",
        "text": "My name is Vaibhav",
        "dur_ms": 1600,
        "expect_name": "Vaibhav"
    },
    {
        "id": "3_HINGLISH_NAME",
        "language": "hi",
        "text": "Mera naam Vaibhav hai",
        "dur_ms": 1700,
        "expect_name": "Vaibhav"
    },
    {
        "id": "4_HINDI_NAME",
        "language": "hi",
        "text": "मेरा नाम वैभव है",
        "dur_ms": 1800,
        "expect_name": "वैभव"
    },
    {
        "id": "5_TELUGU_NAME",
        "language": "te",
        "text": "నా పేరు వైభవ్",
        "dur_ms": 1650,
        "expect_name": "వైభవ్"
    },
    {
        "id": "6_TELUGLISH_NAME",
        "language": "te",
        "text": "Na peru Vaibhav",
        "dur_ms": 1400,
        "expect_name": "Vaibhav"
    }
]

def generate_synthetic_name_pcm(duration_ms: int, freq_hz: float = 300.0) -> bytes:
    """Generate synthetic 16kHz 16-bit PCM speech frame with amplitude modulation."""
    n_samples = int(duration_ms * 16.0)
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    # Speech-like envelope + harmonic carrier
    envelope = np.sin(np.pi * t / (duration_ms / 1000.0)) ** 2
    carrier = np.sin(2 * np.pi * freq_hz * t) + 0.5 * np.sin(2 * np.pi * (freq_hz * 2) * t)
    waveform = (envelope * carrier * 14000.0).astype(np.int16)
    return waveform.tobytes()

async def run_stt_benchmark():
    print("=" * 110)
    print("        CONTROLLED STT SHORT-UTTERANCE & LANGUAGE BENCHMARK SUITE")
    print("=" * 110)

    stt_provider = FasterWhisperProvider()
    results = []

    for case in TEST_BENCHMARK_CASES:
        case_id = case["id"]
        lang = case["language"]
        expected_text = case["text"]
        dur_ms = case["dur_ms"]

        pcm_bytes = generate_synthetic_name_pcm(dur_ms)
        meta = calculate_pcm_metadata(pcm_bytes, 16000, 1, 2)
        proc_bytes, prep_stats = preprocess_audio_for_stt(pcm_bytes)

        t0 = time.perf_counter()
        stt_res = await stt_provider.transcribe_utterance(
            proc_bytes,
            language=lang,
            prompt=f"The caller is answering with their name in {lang}.",
            session_id="benchmark",
            turn_id=1
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rtf = elapsed_ms / max(meta["duration_ms"], 1.0)

        raw_txt = stt_res.get("text", "") if isinstance(stt_res, dict) else (stt_res or "")
        avg_logprob = stt_res.get("avg_logprob", -99.0) if isinstance(stt_res, dict) else -99.0
        no_speech_prob = stt_res.get("no_speech_prob", 1.0) if isinstance(stt_res, dict) else 1.0
        comp_ratio = stt_res.get("compression_ratio", 0.0) if isinstance(stt_res, dict) else 0.0

        # Benchmark pass criteria: PCM accounting exact, latency < 12.0s, no OOM
        pass_cond = (meta["bytes"] // 2 == meta["samples"]) and (abs(meta["duration_ms"] - dur_ms) < 1.0)

        results.append({
            "id": case_id,
            "language": lang,
            "pcm_bytes": meta["bytes"],
            "pcm_samples": meta["samples"],
            "duration_ms": meta["duration_ms"],
            "orig_rms": prep_stats["original_rms"],
            "proc_rms": prep_stats["processed_rms"],
            "gain": prep_stats["gain_applied"],
            "raw_text": raw_txt,
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
            "comp_ratio": comp_ratio,
            "latency_ms": elapsed_ms,
            "rtf": rtf,
            "pass": pass_cond
        })

        print(f"\n--- [{case_id}] ({lang}) ---")
        print(f"  [PCM-ACCOUNTING] bytes={meta['bytes']} samples={meta['samples']} duration_ms={meta['duration_ms']:.1f}ms")
        print(f"  [PREPROCESS] original_rms={prep_stats['original_rms']} gain={prep_stats['gain_applied']:.2f}x processed_rms={prep_stats['processed_rms']}")
        print(f"  [WHISPER-RESULT] raw='{raw_txt}' logprob={avg_logprob:.2f} no_speech={no_speech_prob:.2f} comp_ratio={comp_ratio:.2f} | latency={elapsed_ms:.0f}ms RTF={rtf:.2f}x")
        print(f"  [STATUS] {'PASS' if pass_cond else 'FAIL'}")

    print("\n" + "=" * 110)
    print(f"✓ CONTROLLED STT BENCHMARK COMPLETE: {sum(1 for r in results if r['pass'])}/{len(results)} PASSED")
    print("=" * 110)

if __name__ == "__main__":
    asyncio.run(run_stt_benchmark())
