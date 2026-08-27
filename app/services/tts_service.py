import os
import io
import time
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List
from app.core.logging import logger
from app.core.config import settings
from app.services.speech.tts.base import TextToSpeechProvider
from app.services.speech.tts.svara_provider import SvaraProvider
from app.services.speech.tts.kokoro_provider import KokoroProvider


def get_voice_provider() -> TextToSpeechProvider:
    """Factory function to retrieve the configured TextToSpeechProvider implementation."""
    provider_name = getattr(settings, "TTS_PROVIDER", "svara").lower().strip()
    if provider_name == "svara":
        return SvaraProvider()
    elif provider_name == "kokoro":
        return KokoroProvider()
    else:
        logger.warning(f"Unknown TTS provider '{provider_name}'. Falling back to SvaraProvider.")
        return SvaraProvider()


def get_voice_service() -> "VoiceService":
    """Factory function to retrieve the configured VoiceService facade wrapping the TTS provider."""
    return VoiceService(get_voice_provider())


def find_safe_sentence_boundary(
    text: str,
    min_chars: int = 8,
    target_chars: int = 50,
    max_chars: int = 140
) -> int:
    """
    Finds the optimal character boundary index to split streaming text into coherent speech sentences.
    Respects common titles (Dr., Mr., Mrs.) and numbers to avoid premature sentence splitting.
    """
    if len(text) < min_chars:
        return -1

    search_window = text[:max_chars]
    
    # Check for sentence terminators (. ! ? ।)
    terminators = [".", "!", "?", "।"]
    for i in range(len(search_window) - 1, -1, -1):
        char = search_window[i]
        if char in terminators:
            # Prevent splitting on titles like Dr., Mr., Mrs.
            prefix = search_window[:i].lower()
            if prefix.endswith(("dr", "mr", "mrs", "ms", "prof", "st", "vs")):
                continue
            # Prevent splitting on numbers (e.g. 11.30 AM)
            if i > 0 and i < len(search_window) - 1 and search_window[i-1].isdigit() and search_window[i+1].isdigit():
                continue
            if i >= min_chars:
                return i

    # Fallback to clause boundaries (commas, colons, semicolons) if target length reached
    if len(text) >= target_chars:
        for i in range(min(len(text), max_chars) - 1, min_chars, -1):
            if text[i] in (",", ";", ":", "-", "—"):
                return i

    # Force split at nearest space if buffer exceeds max_chars
    if len(text) >= max_chars:
        space_idx = text.rfind(" ", min_chars, max_chars)
        if space_idx != -1:
            return space_idx
        return max_chars - 1

    return -1


def sanitize_for_tts(text: str, persona: Optional[str] = None) -> str:
    """Sanitizes text output and enforces persona identity before handing to TTS engine."""
    import re
    if not text:
        return ""
    # Strict TTS input validation: reject metadata blocks (Part 16)
    if "[" in text or "]" in text or "{" in text or "}" in text or "customer_name=" in text or "intent=" in text:
        logger.error(f"[TTS-SAFETY-SANITIZER-REJECT] Metadata leak detected in TTS text: '{text}'")
        raise ValueError(f"Metadata leak detected in TTS text: {text}")

    text = text.replace("**", "").replace("*", "")
    text = re.sub(r'#+\s+', '', text)
    text = text.replace("`", "")
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    
    if persona:
        persona_title = persona.title()
        other_personas = [p for p in ["Sophia", "Maya", "Ananya", "Arjun", "David"] if p.lower() != persona.lower()]
        for other in other_personas:
            if other in text and persona_title not in text:
                logger.warning(f"[PERSONA-PIPELINE-SANATIZE] Replacing '{other}' with requested persona '{persona_title}' BEFORE TTS")
                text = text.replace(other, persona_title)

    return text.strip()


class VoiceService:
    """
    High-level facade for text-to-speech synthesis and progressive streaming.
    Instantiates configured provider singleton and manages progressive sentence chunking.
    """

    def __init__(self, provider: Optional[TextToSpeechProvider] = None) -> None:
        self.provider = provider or get_voice_provider()

    @classmethod
    async def warmup(cls) -> float:
        """Warms up the underlying TTS provider model."""
        provider = get_voice_provider()
        if hasattr(provider, "warmup"):
            return await provider.warmup()
        return 0.0

    async def stream_speech(
        self,
        text: str,
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None
    ) -> AsyncGenerator[bytes, None]:
        """Direct pass-through streaming to the underlying TTS provider."""
        async for chunk in self.provider.stream_speech(
            text,
            cancel_event=cancel_event,
            language=language,
            voice_config=voice_config
        ):
            yield chunk

    async def stream_text_stream_progressive(
        self,
        text_generator: AsyncGenerator[str, None],
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Progressive LLM Token -> Sentence Splitter -> Sequenced Parallel Synthesis -> Ordered Playout.
        Guarantees strict sentence playback sequence (1 -> 2 -> 3 -> 4) while pre-synthesizing upcoming sentences.
        """
        sentence_index = 0
        
        # Sequenced Playout Buffer: stores chunks per sentence sequence index
        # sentence_chunks[seq_id] = list of audio bytes
        # sentence_done[seq_id] = bool
        sentence_events: Dict[int, asyncio.Event] = {}
        sentence_chunks: Dict[int, List[bytes]] = {}
        sentence_done: Dict[int, bool] = {}
        
        producer_done = asyncio.Event()

        # Telemetry trackers per sentence ID
        synth_start_times: Dict[int, float] = {}
        synth_end_times: Dict[int, float] = {}
        play_start_times: Dict[int, float] = {}
        play_end_times: Dict[int, float] = {}

        # 1. Text extractor producer: extracts sentence boundaries from LLM token stream
        async def producer():
            nonlocal sentence_index
            buffer = ""
            persona_name = (voice_config or {}).get("persona_name", "Sophia")
            try:
                async for chunk in text_generator:
                    if cancel_event and cancel_event.is_set():
                        break
                    buffer += chunk

                    while True:
                        boundary = find_safe_sentence_boundary(buffer, min_chars=8, target_chars=50, max_chars=140)
                        if boundary == -1:
                            break

                        sentence = sanitize_for_tts(buffer[:boundary + 1], persona=persona_name)
                        buffer = buffer[boundary + 1:]

                        if sentence:
                            sentence_index += 1
                            seq_id = sentence_index
                            sentence_events[seq_id] = asyncio.Event()
                            sentence_chunks[seq_id] = []
                            sentence_done[seq_id] = False
                            logger.info(f"[TTS-QUEUE] sentence={seq_id} text_chars={len(sentence)} text='{sentence}'")
                            asyncio.create_task(_synth_sentence(seq_id, sentence))

                remaining = sanitize_for_tts(buffer, persona=persona_name)
                if remaining and (not cancel_event or not cancel_event.is_set()):
                    sentence_index += 1
                    seq_id = sentence_index
                    sentence_events[seq_id] = asyncio.Event()
                    sentence_chunks[seq_id] = []
                    sentence_done[seq_id] = False
                    logger.info(f"[TTS-QUEUE] sentence={seq_id} (FINAL) text_chars={len(remaining)} text='{remaining}'")
                    asyncio.create_task(_synth_sentence(seq_id, remaining))

            except Exception as e:
                logger.error(f"[TTS-PRODUCER] Text extractor producer failed: {e}")
            finally:
                producer_done.set()

        prefetch_sem = asyncio.Semaphore(4)

        # 2. Worker task: synthesizes speech for a given sentence sequence ID
        async def _synth_sentence(seq_id: int, sentence: str):
            async with prefetch_sem:
                voice_name = (voice_config or {}).get("persona_name", "Sophia")
                lang_code = language or "en"
                logger.info(f"[TTS-SYNTH-START] sentence={seq_id} voice='{voice_name}' lang='{lang_code}' text='{sentence}'")
                t_synth_start = time.perf_counter()
                synth_start_times[seq_id] = t_synth_start
                chunk_count = 0
                try:
                    async for audio_chunk in self.stream_speech(
                        sentence,
                        cancel_event=cancel_event,
                        language=language,
                        voice_config=voice_config
                    ):
                        if cancel_event and cancel_event.is_set():
                            break
                        chunk_count += 1
                        sentence_chunks[seq_id].append(audio_chunk)
                        sentence_events[seq_id].set()

                    inf_ms = (time.perf_counter() - t_synth_start) * 1000.0
                    synth_end_times[seq_id] = time.perf_counter()
                    logger.info(f"[TTS-SYNTH-END] sentence={seq_id} inference_ms={inf_ms:.1f}ms chunks={chunk_count}")
                except Exception as e:
                    logger.error(f"[TTS-WORKER] Synthesis failed for sentence #{seq_id}: {e}")
                finally:
                    sentence_done[seq_id] = True
                    sentence_events[seq_id].set()

        # Start producer task
        producer_task = asyncio.create_task(producer())

        # 3. Sequenced Playout Consumer: yields audio chunks strictly in sentence order 1 -> 2 -> 3
        next_seq_to_play = 1
        previous_chunk_end_t = 0.0

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    break

                # Check if producer is done and all sequence items played out
                if producer_done.is_set() and next_seq_to_play > sentence_index:
                    break

                # Wait for the next sequence ID to be registered
                if next_seq_to_play not in sentence_events:
                    if producer_done.is_set() and next_seq_to_play > sentence_index:
                        break
                    await asyncio.sleep(0.005)
                    continue

                # Wait until sentence_1 has at least 1 audio chunk synthesized
                await sentence_events[next_seq_to_play].wait()

                seq_id = next_seq_to_play
                play_start_t = time.perf_counter()
                play_start_times[seq_id] = play_start_t
                sentence_audio_bytes = 0
                chunk_ptr = 0
                logger.info(f"[TTS-PLAY-START] sentence={seq_id}")

                while True:
                    if cancel_event and cancel_event.is_set():
                        break

                    chunks = sentence_chunks.get(seq_id, [])
                    while chunk_ptr < len(chunks):
                        c = chunks[chunk_ptr]
                        chunk_ptr += 1
                        sentence_audio_bytes += len(c)
                        yield c

                    if sentence_done.get(seq_id, False) and chunk_ptr >= len(sentence_chunks.get(seq_id, [])):
                        break

                    # Wait for next chunk of current sentence
                    await asyncio.sleep(0.002)

                play_end_t = time.perf_counter()
                play_end_times[seq_id] = play_end_t
                play_dur_ms = (play_end_t - play_start_t) * 1000.0
                audio_duration_ms = sentence_audio_bytes / 48.0
                
                # Calculate playback gap telemetry metrics (Requirement 24)
                gen_start = synth_start_times.get(seq_id, play_start_t)
                gen_end = synth_end_times.get(seq_id, play_start_t)
                synthesis_wait_ms = (gen_end - gen_start) * 1000.0
                queue_wait_ms = (play_start_t - gen_end) * 1000.0 if gen_end > 0 else 0.0
                playback_gap_ms = (play_start_t - previous_chunk_end_t) * 1000.0 if previous_chunk_end_t > 0.0 else 0.0
                previous_chunk_end_t = play_end_t

                # [TTS-FLOW] Mandatory Telemetry Logging (Requirement 24)
                logger.info(
                    f"[TTS-FLOW] sentence_id={seq_id} generation_start={gen_start:.3f} generation_end={gen_end:.3f} "
                    f"playback_start={play_start_t:.3f} playback_end={play_end_t:.3f} audio_duration={audio_duration_ms:.1f}ms "
                    f"queue_wait_ms={queue_wait_ms:.1f}ms synthesis_wait_ms={synthesis_wait_ms:.1f}ms "
                    f"playback_gap_ms={playback_gap_ms:.1f}ms previous_chunk_end={previous_chunk_end_t:.3f} "
                    f"current_chunk_start={play_start_t:.3f} gap_ms={playback_gap_ms:.1f}ms"
                )

                if playback_gap_ms > 100.0:
                    logger.error(f"[TTS-FLOW-GAP-ERROR] Excessive inter-sentence gap! gap_ms={playback_gap_ms:.1f}ms (>100ms threshold)")
                elif playback_gap_ms > 30.0:
                    logger.warning(f"[TTS-FLOW-GAP-WARN] Noticeable inter-sentence gap! gap_ms={playback_gap_ms:.1f}ms (>30ms threshold)")

                next_seq_to_play += 1

        finally:
            producer_task.cancel()
