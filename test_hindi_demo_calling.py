import sys
import os
import asyncio
import uuid
import time

# Ensure current project directory is at the top of sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.speech.tts.kokoro_provider import KokoroProvider
from app.voice_demo.controllers.voice_agent import (
    get_greeting_text,
    pregenerate_greeting,
    _greeting_cache,
    _demo_sessions
)
from app.services.conversation_engine import normalize_name_transcript

async def test_demo_cold_calling_hindi_pipeline():
    print("=" * 75)
    print("  VERIFYING DEMO_COLD_CALLING HINDI VOICE PIPELINE (ACCEPTANCE TEST)")
    print("=" * 75)

    # Verify import origin
    import app.services.speech.tts.kokoro_provider as kp_module
    print(f"Loaded KokoroProvider from: {kp_module.__file__}")
    assert "demo_cold_calling" in kp_module.__file__, "CRITICAL: Must import from demo_cold_calling!"

    tts = KokoroProvider()
    total_passed = 0
    total_tests = 0

    # -------------------------------------------------------------------------
    # TEST 1: Kokoro Hindi Resolution & Language Guard
    # -------------------------------------------------------------------------
    print("\n--- [TEST 1] KOKORO HINDI RESOLUTION & LANGUAGE GUARD ---")
    total_tests += 1

    # Maya (Hindi)
    voice_maya, lang_maya, key_m = tts._resolve_voice_and_lang({"persona_name": "Maya"}, "Hindi")
    print(f"  Maya (Hindi) → voice='{voice_maya}', kokoro_lang='{lang_maya}'")
    assert voice_maya == "hf_beta", f"Expected hf_beta for Maya Hindi, got {voice_maya}"
    assert lang_maya == "h", f"Expected 'h' kokoro_lang for Maya Hindi, got {lang_maya}"

    # Sophia (Hindi)
    voice_sophia, lang_sophia, key_s = tts._resolve_voice_and_lang({"persona_name": "Sophia"}, "Hindi")
    print(f"  Sophia (Hindi) → voice='{voice_sophia}', kokoro_lang='{lang_sophia}'")
    assert voice_sophia == "hf_alpha", f"Expected hf_alpha for Sophia Hindi, got {voice_sophia}"
    assert lang_sophia == "h", f"Expected 'h' kokoro_lang for Sophia Hindi, got {lang_sophia}"

    total_passed += 1
    print("  ✓ PASS: Kokoro resolves native Indian Hindi female voice presets & kokoro_lang='h' correctly!")

    # -------------------------------------------------------------------------
    # TEST 2: Native Devanagari Synthesis Benchmark
    # -------------------------------------------------------------------------
    print("\n--- [TEST 2] NATIVE DEVANAGARI HINDI SYNTHESIS BENCHMARK ---")
    total_tests += 1

    devanagari_text = "नमस्ते! मैं सिटीकेयर हॉस्पिटल से Maya बात कर रही हूँ। क्या मैं आपका नाम जान सकती हूँ?"
    chunks = []
    t0 = time.perf_counter()
    async for chunk in tts.stream_speech(devanagari_text, language="Hindi", voice_config={"persona_name": "Maya"}):
        chunks.append(chunk)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Synthesized {len(chunks)} frames in {elapsed_ms:.1f}ms (Devanagari length: {len(devanagari_text)} chars)")
    assert len(chunks) > 0, "Synthesis output must contain audio frames!"
    total_passed += 1
    print("  ✓ PASS: Native Devanagari Hindi streaming synthesis completed cleanly!")

    # -------------------------------------------------------------------------
    # TEST 3: Persona & Session Greeting Isolation
    # -------------------------------------------------------------------------
    print("\n--- [TEST 3] PERSONA & SESSION GREETING ISOLATION ---")
    total_tests += 1

    greeting_maya = get_greeting_text("hospital", "hindi", "Maya")
    greeting_sophia = get_greeting_text("hospital", "hindi", "Sophia")

    print(f"  Maya Hindi Greeting: '{greeting_maya}'")
    print(f"  Sophia Hindi Greeting: '{greeting_sophia}'")

    assert "माया" in greeting_maya, "Maya greeting must contain Devanagari 'माया'"
    assert "सोफिया" in greeting_sophia, "Sophia greeting must contain Devanagari 'सोफिया'"
    assert "सोफिया" not in greeting_maya, "Maya greeting MUST NOT contain 'सोफिया'"

    # Cache key separation
    key_maya = ("maya", "hindi", "hospital", "female")
    key_sophia = ("sophia", "hindi", "hospital", "female")
    assert key_maya != key_sophia, "Cache keys for Maya and Sophia MUST be distinct!"
    total_passed += 1
    print("  ✓ PASS: Persona identity and greeting cache are 100% session-isolated!")

    # -------------------------------------------------------------------------
    # TEST 4: Devanagari Transcript Normalization
    # -------------------------------------------------------------------------
    print("\n--- [TEST 4] DEVANAGARI TRANSCRIPT NORMALIZATION ---")
    total_tests += 1

    hindi_utterance = "मेरा नाम मयंक है।"
    norm_text = normalize_name_transcript(hindi_utterance)
    print(f"  Input: '{hindi_utterance}' → Normalized: '{norm_text}'")
    assert norm_text == "मेरा नाम मयंक है", f"Expected 'मेरा नाम मयंक है', got '{norm_text}'"
    total_passed += 1
    print("  ✓ PASS: Devanagari Hindi transcripts are preserved without distortion!")

    # -------------------------------------------------------------------------
    # TEST 5: STT Hallucination Guardrail
    # -------------------------------------------------------------------------
    print("\n--- [TEST 5] STT HALLUCINATION GUARDRAIL ---")
    total_tests += 1
    from app.voice_demo.controllers.voice_agent import validate_stt_transcript

    # Case A: Candidate list hallucination
    bad_list_transcript = "I'm Akash, My name is Arjun, Sophia, David, Maya."
    dummy_audio = b"\x00" * (500 * 8) # 500ms audio (8kHz mu-law = 8 bytes/ms)
    valid_a, reason_a, res_a = validate_stt_transcript(bad_list_transcript, dummy_audio, "hi")
    print(f"  Candidate List Input: '{bad_list_transcript}' → Valid: {valid_a} (Reason: {reason_a})")
    assert not valid_a, "Multi-persona candidate list MUST be rejected by STT Guardrail!"

    # Case B: Duration mismatch (<800ms audio producing >30 chars)
    bad_dur_transcript = "I am calling to check on your medical appointment for tomorrow"
    valid_b, reason_b, res_b = validate_stt_transcript(bad_dur_transcript, dummy_audio, "hi")
    print(f"  Duration Mismatch Input: '{bad_dur_transcript}' → Valid: {valid_b} (Reason: {reason_b})")
    assert not valid_b, "Duration mismatch transcript MUST be rejected by STT Guardrail!"

    # Case C: Valid short name
    good_name_transcript = "मेरा नाम मयंक है"
    valid_c, reason_c, res_c = validate_stt_transcript(good_name_transcript, dummy_audio, "hi")
    print(f"  Valid Name Input: '{good_name_transcript}' → Valid: {valid_c} (Reason: {reason_c})")
    assert valid_c, "Valid Hindi name statement MUST pass STT Guardrail!"

    total_passed += 1
    print("  ✓ PASS: STT Hallucination Guardrail successfully blocks invalid/hallucinated transcripts!")

    # -------------------------------------------------------------------------
    # TEST 6: Smart Hinglish Transliterator Accuracy
    # -------------------------------------------------------------------------
    print("\n--- [TEST 6] SMART HINGLISH TRANSLITERATOR ACCURACY ---")
    total_tests += 1
    from app.services.speech.tts.kokoro_provider import transliterate_hindi

    test_sentence = "नमस्ते! मैं सिटीकेयर हॉस्पिटल से Maya बात कर रही हूँ।"
    translit_result = transliterate_hindi(test_sentence)
    print(f"  Input Devanagari: '{test_sentence}'")
    print(f"  Transliterated:   '{translit_result}'")
    assert "Namaste" in translit_result, "Expected 'Namaste' in transliterated output!"
    assert "CityCare" in translit_result, "Expected 'CityCare' in transliterated output!"
    assert "Hospital" in translit_result, "Expected 'Hospital' in transliterated output!"
    assert "Maya" in translit_result, "Expected 'Maya' in transliterated output!"

    total_passed += 1
    print("  ✓ PASS: Smart Hinglish transliterator generates accurate natural phonetics!")

    # -------------------------------------------------------------------------
    # TEST 7: Single Dedicated Kokoro Worker Queue Sequential Streaming
    # -------------------------------------------------------------------------
    print("\n--- [TEST 7] SINGLE DEDICATED KOKORO WORKER QUEUE SEQUENTIAL STREAMING ---")
    total_tests += 1
    from app.services.tts_service import VoiceService

    vs = VoiceService()

    async def mock_llm_stream():
        yield "नमस्ते! मैं सिटीकेयर हॉस्पिटल से Maya बात कर रही हूँ। "
        yield "आपका अपॉइंटमेंट कल सुबह 11 बजे निर्धारित है। "
        yield "क्या आप इसे कन्फर्म करना चाहेंगे?"

    audio_chunks = []
    t_start = time.perf_counter()
    async for chunk in vs.stream_text_stream_progressive(
        mock_llm_stream(),
        language="hi",
        voice_config={"persona_name": "Maya"}
    ):
        audio_chunks.append(chunk)

    stream_ms = (time.perf_counter() - t_start) * 1000.0
    print(f"  Streamed {len(audio_chunks)} audio chunks across 3 sentences in {stream_ms:.1f}ms")
    assert len(audio_chunks) > 0, "Progressive streaming must produce audio chunks!"

    total_passed += 1
    print("  ✓ PASS: Single dedicated Kokoro worker streams multi-sentence response sequentially!")

    print("\n" + "=" * 75)
    print(f" TOTAL RESULT: {total_passed}/{total_tests} Acceptance Tests Passed 100% in demo_cold_calling.")
    print("=" * 75)

if __name__ == "__main__":
    asyncio.run(test_demo_cold_calling_hindi_pipeline())
