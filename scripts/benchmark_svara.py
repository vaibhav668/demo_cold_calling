import os
import sys
import time
import asyncio
import psutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.config import settings
from app.core.logging import logger
from app.services.speech.tts.svara_provider import SvaraProvider

TEST_SENTENCES = {
    "English": [
        "Hi, this is Maya from CityCare Hospital.",
        "May I know whom I'm speaking with?",
        "Your appointment is confirmed for tomorrow at 11 AM.",
        "Thank you for your time. Have a wonderful day."
    ],
    "Hindi": [
        "नमस्ते, मैं सिटीकेयर हॉस्पिटल से माया बात कर रही हूँ।",
        "क्या मैं आपका नाम जान सकती हूँ?",
        "आपकी अपॉइंटमेंट कल सुबह 11 बजे डॉक्टर शर्मा के साथ निर्धारित है।",
        "आपकी अपॉइंटमेंट सफलतापूर्वक कन्फर्म हो गई है।",
        "आपका समय देने के लिए धन्यवाद। आपका दिन शुभ हो।"
    ],
    "Telugu": [
        "నమస్కారం, నేను సిటీకేర్ హాస్పిటల్ నుండి మాయ మాట్లాడుతున్నాను.",
        "మీ పేరు తెలుసుకోవచ్చా?",
        "మీ అపాయింట్మెంట్ రేపు ఉదయం 11 గంటలకు డాక్టర్ శర్మతో ఉంది.",
        "మీ అపాయింట్మెంట్ విజయవంతంగా నిర్ధారించబడింది.",
        "మీ సమయానికి ధన్యవాదాలు. మీ రోజు శుభంగా ఉండాలి."
    ]
}


async def run_benchmark():
    process = psutil.Process()
    ram_startup = process.memory_info().rss / (1024 * 1024)
    print("=" * 60)
    print("        SVARA CPU BENCHMARK - DEMO COLD CALLING")
    print("=" * 60)
    print(f"Startup RAM: {ram_startup:.2f} MB")

    # 1. Model Loading & Warmup
    t0 = time.perf_counter()
    provider = SvaraProvider()
    warmup_ms = await SvaraProvider.warmup()
    load_time_sec = time.perf_counter() - t0
    ram_loaded = process.memory_info().rss / (1024 * 1024)

    print(f"Model Load & Warmup Time: {load_time_sec:.2f}s ({warmup_ms:.1f}ms)")
    print(f"Svara Loaded RAM: {ram_loaded:.2f} MB")
    print("-" * 60)

    results = []

    # 2. Test Execution
    for lang, sentences in TEST_SENTENCES.items():
        lang_code = {"English": "en", "Hindi": "hi", "Telugu": "te"}[lang]
        persona = "Maya"

        print(f"\n--- Testing Language: {lang} (Persona: {persona}) ---")
        for idx, text in enumerate(sentences, 1):
            t_gen_start = time.perf_counter()
            chunks = []
            ttfb_ms = 0.0

            async for chunk in provider.stream_speech(
                text,
                language=lang_code,
                voice_config={"persona_name": persona}
            ):
                if not chunks:
                    ttfb_ms = (time.perf_counter() - t_gen_start) * 1000.0
                chunks.append(chunk)

            total_gen_sec = time.perf_counter() - t_gen_start
            total_bytes = sum(len(c) for c in chunks)
            audio_duration_sec = total_bytes / 8000.0  # 8kHz G.711 mu-law = 8000 B/s
            rtf = total_gen_sec / max(0.001, audio_duration_sec)

            results.append({
                "language": lang,
                "sentence_id": idx,
                "chars": len(text),
                "ttfb_ms": ttfb_ms,
                "total_sec": total_gen_sec,
                "audio_sec": audio_duration_sec,
                "rtf": rtf
            })

            print(
                f"  [{lang} #{idx}] TTFB: {ttfb_ms:6.1f}ms | Total: {total_gen_sec:5.2f}s | "
                f"Audio: {audio_duration_sec:5.2f}s | RTF: {rtf:4.2f}x | text='{text[:30]}...'"
            )

    # 3. Benchmark Summary Table
    print("\n" + "=" * 60)
    print("                   SVARA CPU BENCHMARK REPORT")
    print("=" * 60)
    print(f"{'Language':<12} | {'TTFB (ms)':<10} | {'Total (s)':<10} | {'Audio (s)':<10} | {'RTF':<6}")
    print("-" * 60)

    for lang in ["English", "Hindi", "Telugu"]:
        lang_res = [r for r in results if r["language"] == lang]
        if lang_res:
            avg_ttfb = sum(r["ttfb_ms"] for r in lang_res) / len(lang_res)
            avg_total = sum(r["total_sec"] for r in lang_res) / len(lang_res)
            avg_audio = sum(r["audio_sec"] for r in lang_res) / len(lang_res)
            avg_rtf = sum(r["rtf"] for r in lang_res) / len(lang_res)
            print(f"{lang:<12} | {avg_ttfb:10.1f} | {avg_total:10.2f} | {avg_audio:10.2f} | {avg_rtf:6.2f}x")

    cpu_usage = psutil.cpu_percent(interval=1.0)
    print("-" * 60)
    print(f"RAM: Startup: {ram_startup:.1f} MB | Svara Loaded: {ram_loaded:.1f} MB")
    print(f"CPU Utilization: Average ~{cpu_usage:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
