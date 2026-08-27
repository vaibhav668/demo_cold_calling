import sys
import os
import time
import asyncio
import wave
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.config import settings
from app.services.speech.tts.svara_provider import SvaraProvider, trim_pcm_digital_silence

TELUGU_BENCHMARK_SCENARIOS = [
    ("1_greeting", "Telugu", "Maya", "నమస్కారం! నేను మాయా మాట్లాడుతున్నాను, CityCare Hospital నుంచి కాల్ చేస్తున్నాను.", "GREETING"),
    ("1_greeting_teluglish", "Telugu", "Maya", "Namaskaram! Nenu Maya matladutunnanu, CityCare Hospital nundi call chestunnanu.", "GREETING"),
    
    ("2_asking_name", "Telugu", "Maya", "మీ పేరు తెలుసుకోవచ్చా?", "FRIENDLY_QUESTION"),
    ("2_asking_name_teluglish", "Telugu", "Maya", "Mee peru telusukovachha?", "FRIENDLY_QUESTION"),
    
    ("3_acknowledging_name", "Telugu", "Maya", "మీతో మాట్లాడటం చాలా సంతోషంగా ఉంది, Vaibhav గారు!", "POSITIVE"),
    
    ("4_appointment_info", "Telugu", "Maya", "మీకు రేపు ఉదయం 11 AM కి Dr. Sharma గారితో appointment ఉంది కదా?", "PURPOSE"),
    
    ("5_confirmation", "Telugu", "Maya", "చాలా మంచిది, Vaibhav గారు! మీ appointment రేపు ఉదయం 11 గంటలకు confirm అయింది.", "CONFIRMATION"),
    
    ("6_cancellation", "Telugu", "Maya", "సరే, Vaibhav గారు. మీ appointment successfully cancel చేశాను.", "CANCELLATION"),
    
    ("7_rescheduling", "Telugu", "Maya", "చాలా బాగుంది, Vaibhav గారు. మీ appointment వచ్చే సోమవారం మధ్యాహ్నం 2 PM కి reschedule చేశాను.", "RESCHEDULING"),
    
    ("8_closing", "Telugu", "Maya", "మీ సమయానికి చాలా ధన్యవాదాలు, Vaibhav గారు! మంచి రోజు అవ్వాలని కోరుకుంటున్నాను! Bye!", "CLOSING"),
]

async def run_telugu_benchmark():
    out_dir = os.path.abspath("test_telugu_voice_outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 95)
    print("      TELUGU HUMAN-LIKE EXPRESSIVE SPEECH & ZERO-GAP DIAGNOSTIC BENCHMARK")
    print("=" * 95)

    svara = SvaraProvider()
    results = []

    for scenario_id, lang, persona, text, intent in TELUGU_BENCHMARK_SCENARIOS:
        fname = f"{scenario_id}_{persona.lower()}.wav"
        fpath = os.path.join(out_dir, fname)

        t_start = time.perf_counter()
        pcm_chunks = []
        voice_config = {"persona_name": persona, "intent": intent, "session_id": "telugu_benchmark"}

        async for chunk in svara.stream_speech(text, language=lang, voice_config=voice_config):
            pcm_chunks.append(chunk)

        total_pcm = b"".join(pcm_chunks)
        synth_time = (time.perf_counter() - t_start) * 1000.0
        audio_dur_sec = len(total_pcm) / (24000 * 2) if total_pcm else 0.0
        rtf = (synth_time / 1000.0) / audio_dur_sec if audio_dur_sec > 0 else 0.0

        # Calculate silence trimming metrics
        trimmed_pcm, leading_ms, trailing_ms = trim_pcm_digital_silence(total_pcm, sample_rate=24000, pad_ms=10)

        chunk_count = len(pcm_chunks)
        avg_chunk_dur_ms = (len(total_pcm) / chunk_count / 48.0) if chunk_count > 0 else 0.0

        if trimmed_pcm:
            with wave.open(fpath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(trimmed_pcm)

        results.append({
            "scenario": scenario_id,
            "dur_sec": audio_dur_sec,
            "synth_ms": synth_time,
            "rtf": rtf,
            "leading_silence_ms": leading_ms,
            "trailing_silence_ms": trailing_ms,
            "chunk_count": chunk_count,
            "avg_chunk_ms": avg_chunk_dur_ms,
            "file": fname
        })

        print(
            f"[{scenario_id:25}] dur={audio_dur_sec:5.2f}s synth={synth_time:6.1f}ms RTF={rtf:4.2f}x "
            f"lead_silence={leading_ms:4.1f}ms trail_silence={trailing_ms:4.1f}ms chunks={chunk_count:2d} -> {fname}"
        )

    print("=" * 95)
    print(f"✓ GENERATED {len(results)} TELUGU AUDIO SAMPLES IN: {out_dir}")
    print("=" * 95)

if __name__ == "__main__":
    asyncio.run(run_telugu_benchmark())
