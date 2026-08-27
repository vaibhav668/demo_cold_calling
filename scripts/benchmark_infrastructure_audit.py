import sys
import os
import time
import asyncio
import psutil
import wave
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.config import settings

def get_process_rss_mb() -> float:
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)

async def run_benchmark():
    print("=" * 80)
    print("      INFRASTRUCTURE & RESOURCE BENCHMARK (NON-DESTRUCTIVE)")
    print("=" * 80)

    # 1. Idle RSS
    rss_idle = get_process_rss_mb()
    print(f"[STAGE 1] Base Python Idle RSS: {rss_idle:.2f} MB")

    # 2. After VAD load
    t0 = time.perf_counter()
    from app.services.speech.vad.silero_provider import SileroVADProvider
    vad = SileroVADProvider()
    vad.process_frame(b"\x00" * 160)
    rss_vad = get_process_rss_mb()
    t_vad = (time.perf_counter() - t0) * 1000.0
    print(f"[STAGE 2] After Silero VAD load: {rss_vad:.2f} MB (Delta: +{rss_vad - rss_idle:.2f} MB, Load: {t_vad:.1f}ms)")

    # 3. After Whisper STT load
    t0 = time.perf_counter()
    from app.services.speech.stt.faster_whisper_provider import FasterWhisperProvider
    stt = FasterWhisperProvider()
    await FasterWhisperProvider.warmup(model_size=settings.WHISPER_MODEL)
    rss_stt = get_process_rss_mb()
    t_stt = (time.perf_counter() - t0) * 1000.0
    print(f"[STAGE 3] After Faster-Whisper STT load ({settings.WHISPER_MODEL}): {rss_stt:.2f} MB (Delta: +{rss_stt - rss_vad:.2f} MB, Load: {t_stt:.1f}ms)", flush=True)

    # 4. After Svara TTS load
    t0 = time.perf_counter()
    from app.services.speech.tts.svara_provider import SvaraProvider
    tts = SvaraProvider()
    await SvaraProvider.warmup()
    rss_tts = get_process_rss_mb()
    t_tts = (time.perf_counter() - t0) * 1000.0
    print(f"[STAGE 4] After Svara TTS load: {rss_tts:.2f} MB (Delta: +{rss_tts - rss_stt:.2f} MB, Load: {t_tts:.1f}ms)")

    # 5. Measure Single TTS Inference (Peak RSS, CPU, RTF, Latency)
    print("\n--- SINGLE CALL BENCHMARKS ---")
    test_phrase = "Hi, this is Maya from CityCare Hospital. I'm calling to confirm your appointment with Dr. Sharma tomorrow at 11 AM."
    
    cpu_before = psutil.cpu_percent(interval=None)
    t_tts_start = time.perf_counter()
    pcm_chunks = []
    async for chunk in tts.stream_speech(test_phrase, language="en", voice_config={"persona_name": "Maya", "intent": "GREETING"}):
        pcm_chunks.append(chunk)
    
    t_tts_end = time.perf_counter()
    cpu_during_tts = psutil.cpu_percent(interval=0.1)
    rss_tts_peak = get_process_rss_mb()
    
    total_pcm = b"".join(pcm_chunks)
    audio_dur_sec = len(total_pcm) / (24000 * 2) if total_pcm else 0.0
    tts_synth_ms = (t_tts_end - t_tts_start) * 1000.0
    tts_rtf = (tts_synth_ms / 1000.0) / audio_dur_sec if audio_dur_sec > 0 else 0.0

    print(f"[TTS SINGLE] Text length: {len(test_phrase)} chars | Audio duration: {audio_dur_sec:.2f}s")
    print(f"[TTS SINGLE] Synthesis time: {tts_synth_ms:.1f}ms | RTF: {tts_rtf:.2f}x")
    print(f"[TTS SINGLE] Peak RSS: {rss_tts_peak:.2f} MB (Delta: +{rss_tts_peak - rss_tts:.2f} MB)")
    print(f"[TTS SINGLE] CPU Utilization: {cpu_during_tts:.1f}%")

    # 6. Measure Single STT Inference (Whisper local pass)
    # Generate 2 seconds of dummy 8kHz mu-law audio
    import audioop
    pcm_8k_dummy = (np.random.randn(16000) * 1000).astype(np.int16).tobytes()
    ulaw_dummy = audioop.lin2ulaw(pcm_8k_dummy, 2)

    t_stt_start = time.perf_counter()
    cpu_stt_before = psutil.cpu_percent(interval=None)
    stt_res = await stt.transcribe_utterance(ulaw_dummy, language="en")
    t_stt_end = time.perf_counter()
    cpu_during_stt = psutil.cpu_percent(interval=0.1)
    rss_stt_peak = get_process_rss_mb()
    stt_ms = (t_stt_end - t_stt_start) * 1000.0
    stt_rtf = (stt_ms / 1000.0) / 2.0

    print(f"\n[STT SINGLE] 2.0s Audio STT latency: {stt_ms:.1f}ms | RTF: {stt_rtf:.2f}x")
    print(f"[STT SINGLE] Peak RSS: {rss_stt_peak:.2f} MB (Delta: +{rss_stt_peak - rss_tts_peak:.2f} MB)")
    print(f"[STT SINGLE] CPU Utilization: {cpu_during_stt:.1f}%")

    # 7. Concurrency Benchmarks (1, 2, 5 concurrent TTS synthesis calls)
    print("\n--- CONCURRENCY BENCHMARKS (TTS Prefetch & Synthesis) ---")

    async def single_call_task(call_id: int):
        phrase = f"Call {call_id}: Hi this is Maya from CityCare Hospital confirming your appointment for tomorrow."
        t_start = time.perf_counter()
        chunks = []
        async for c in tts.stream_speech(phrase, language="en", voice_config={"persona_name": "Maya", "session_id": f"sim_{call_id}"}):
            chunks.append(c)
        elapsed = (time.perf_counter() - t_start) * 1000.0
        dur = len(b"".join(chunks)) / 48000.0
        return elapsed, dur

    for num_calls in [1, 2, 5]:
        rss_start = get_process_rss_mb()
        t_batch_start = time.perf_counter()
        tasks = [single_call_task(i) for i in range(num_calls)]
        res_list = await asyncio.gather(*tasks)
        t_batch_end = time.perf_counter()
        rss_batch_peak = get_process_rss_mb()
        
        batch_total_ms = (t_batch_end - t_batch_start) * 1000.0
        total_audio_sec = sum(r[1] for r in res_list)
        avg_call_ms = sum(r[0] for r in res_list) / len(res_list)

        print(f"[{num_calls} CONCURRENT CALLS] Total time: {batch_total_ms:.1f}ms | Avg call latency: {avg_call_ms:.1f}ms")
        print(f"[{num_calls} CONCURRENT CALLS] Total audio generated: {total_audio_sec:.2f}s | Peak RSS: {rss_batch_peak:.2f} MB (Memory delta: +{rss_batch_peak - rss_start:.2f} MB)")

    print("\n=" * 80)
    print("      BENCHMARK COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
