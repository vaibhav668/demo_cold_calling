import time
import inspect
import asyncio
import re
from typing import AsyncGenerator, Optional
from app.core.config import settings
from app.services.speech.tts.base import TextToSpeechProvider
from app.services.speech.tts.kokoro_provider import KokoroProvider
from app.services.speech.tts.edge_tts_provider import EdgeTTSProvider
from app.core.logging import logger

_voice_service_instance = None

def get_voice_service() -> "VoiceService":
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = VoiceService()
    return _voice_service_instance


def sanitize_for_tts(text: str) -> str:
    """
    Strips all implementation leaks, internal tokens, XML, JSON, tool call syntax,
    and markdown formatting, leaving only clean conversational text.
    """
    if not text:
        return ""
        
    # 1. Strip markdown code blocks (e.g. ```json ... ```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # 2. Strip JSON-like structures (any braces {...} containing colons, quotes, etc.)
    text = re.sub(r'\{[^{}]*?["\']\s*:\s*[^{}]*?\}', '', text)
    # Also strip general curly braces content if it looks like JSON
    text = re.sub(r'\{[^{}]*?\}', '', text)
    
    # 3. Strip HTML / XML tags (e.g., </function>, <tool_call>, etc.)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 4. Strip tool execution and state tags (e.g. [STATE: ...], [EXTRACT: ...])
    text = re.sub(r'\[STATE:[^\]]*\]', '', text)
    text = re.sub(r'\[EXTRACT:[^\]]*\]', '', text)
    text = re.sub(r'\[RECOVERY_SAY:[^\]]*\]', '', text)
    
    # 5. Remove common tool names / system leakage keywords
    system_patterns = [
        r'(?i)\btransfer_to_human\b',
        r'(?i)\bend_call\b',
        r'(?i)\btool_call\b',
        r'(?i)\bfunction_call\b',
    ]
    for pattern in system_patterns:
        text = re.sub(pattern, '', text)
        
    # 6. Clean markdown symbols
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = re.sub(r'#+\s+', '', text)
    
    # 7. Clean up extra whitespace and residual punctuation
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class VoiceService:
    """
    Service Facade representing the Text-to-Speech (TTS) interface.
    Conversation Engine and Telephony systems use this facade to synthesize audio.
    """

    @classmethod
    async def warmup(cls) -> float:
        """Warms up the voice service by preloading Kokoro model if active."""
        instance = get_voice_service()
        if isinstance(instance.provider, KokoroProvider):
            return await KokoroProvider.warmup()
        return 0.0

    def __init__(self) -> None:
        provider_name = settings.TTS_PROVIDER.lower().strip()
        logger.info(f"[TTS-INIT] Selecting TTS provider: '{provider_name}'")
        
        self.provider = None
        if provider_name == "kokoro":
            try:
                self.provider = KokoroProvider()
            except Exception as e:
                logger.error(f"[TTS-INIT] Kokoro initialization failed, falling back to EdgeTTS: {e}")
                self.provider = EdgeTTSProvider()
        else:
            self.provider = EdgeTTSProvider()

    async def stream_speech(
        self,
        text: str,
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Delegate synthesis streaming to the configured provider (Kokoro or EdgeTTS)."""
        async for chunk in self.provider.stream_speech(
            text, cancel_event=cancel_event, language=language, voice_config=voice_config
        ):
            yield chunk

    async def stream_text_stream_progressive(
        self,
        text_stream: AsyncGenerator[str, None],
        cancel_event: Optional[asyncio.Event] = None,
        language: Optional[str] = None,
        voice_config: Optional[dict] = None,
        is_greeting: bool = False
    ) -> AsyncGenerator[bytes, None]:
        """
        Consumes an incoming LLM token stream, splits into sentences progressively,
        and synthesizes audio chunks using a true producer-consumer model where
        synthesis of sentence i overlaps with the playout of sentence i-1.
        """
        # Natural segmenter limits
        MIN_CHARS = 25
        MAX_CHARS = 50
        punctuation = {'.', '?', '!', '\n'}
        secondary_punctuation = {',', ';', ':', '—', '-'}
        abbreviations = ("dr.", "mr.", "mrs.", "ms.", "vs.", "st.", "co.", "inc.", "ltd.", "e.g.", "i.e.")

        sentence_queues = asyncio.Queue()
        active_tasks = set()

        async def _synthesize_sentence_task(sentence: str, queue: asyncio.Queue):
            try:
                t_synth_start = time.perf_counter()
                async for chunk in self.stream_speech(
                    sentence,
                    cancel_event=cancel_event,
                    language=language,
                    voice_config=voice_config
                ):
                    if cancel_event and cancel_event.is_set():
                        break
                    await queue.put(chunk)
                t_synth_end = time.perf_counter()
                logger.info(f"[TTS-OVERLAP] Synthesized sentence in {(t_synth_end - t_synth_start)*1000:.0f}ms: '{sentence}'")
            except Exception as e:
                logger.error(f"[TTS-OVERLAP] Synthesis failed for sentence '{sentence}': {e}")
            finally:
                await queue.put(None)

        async def producer():
            try:
                buffer = ""
                turn_start = time.perf_counter()
                sentence_index = 0

                async for chunk in text_stream:
                    if cancel_event and cancel_event.is_set():
                        break
                    buffer += chunk

                    while True:
                        positions = sorted(
                            idx
                            for p in punctuation
                            for idx in _find_all(buffer, p)
                        )
                        found_idx = -1
                        if positions:
                            for pos in positions:
                                seg = buffer[:pos + 1]
                                if '\n' in buffer[pos]:
                                    found_idx = pos
                                    break
                                if len(seg.strip()) >= MIN_CHARS:
                                    low_seg = seg.strip().lower()
                                    if not any(low_seg.endswith(abbr) for abbr in abbreviations):
                                        found_idx = pos
                                        break

                        # If no primary punctuation matches, but buffer exceeds MAX_CHARS,
                        # try to split at a secondary punctuation (like comma) or a space to keep chunk size <= 50
                        if found_idx == -1 and len(buffer.strip()) >= MAX_CHARS:
                            sec_positions = sorted(
                                idx
                                for p in secondary_punctuation
                                for idx in _find_all(buffer, p)
                            )
                            for pos in sec_positions:
                                if len(buffer[:pos + 1].strip()) >= MIN_CHARS:
                                    found_idx = pos
                                    break

                            if found_idx == -1:
                                space_positions = list(_find_all(buffer, ' '))
                                for pos in reversed(space_positions):
                                    if MIN_CHARS <= pos <= MAX_CHARS:
                                        found_idx = pos
                                        break

                        if found_idx == -1:
                            break

                        sentence = buffer[:found_idx + 1].strip()
                        buffer = buffer[found_idx + 1:]

                        if sentence:
                            sentence_index += 1
                            logger.info(
                                f"[TTS-SENTENCE #{sentence_index}] Extracted: '{sentence}' "
                                f"| turn_elapsed={(time.perf_counter() - turn_start)*1000:.0f}ms"
                            )
                            sentence_queue = asyncio.Queue()
                            await sentence_queues.put((sentence, sentence_queue))
                            
                            synth_task = asyncio.create_task(
                                _synthesize_sentence_task(sentence, sentence_queue)
                            )
                            active_tasks.add(synth_task)
                            synth_task.add_done_callback(active_tasks.discard)

                # Process leftover buffer
                remaining = buffer.strip()
                if remaining and (not cancel_event or not cancel_event.is_set()):
                    sentence_index += 1
                    logger.info(
                        f"[TTS-SENTENCE #{sentence_index} FINAL] Extracted: '{remaining}' "
                        f"| turn_elapsed={(time.perf_counter() - turn_start)*1000:.0f}ms"
                    )
                    sentence_queue = asyncio.Queue()
                    await sentence_queues.put((remaining, sentence_queue))
                    
                    synth_task = asyncio.create_task(
                        _synthesize_sentence_task(remaining, sentence_queue)
                    )
                    active_tasks.add(synth_task)
                    synth_task.add_done_callback(active_tasks.discard)

            except Exception as e:
                logger.error(f"[TTS-OVERLAP] Producer task failed: {e}")
            finally:
                await sentence_queues.put(None)

        # Start the producer task
        producer_task = asyncio.create_task(producer())

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    break

                queue_item = await sentence_queues.get()
                if queue_item is None:
                    break

                sentence, sentence_queue = queue_item
                logger.info(f"[TTS-OVERLAP] Playout started for sentence: '{sentence}'")

                while True:
                    if cancel_event and cancel_event.is_set():
                        break
                    chunk = await sentence_queue.get()
                    if chunk is None:
                        break
                    yield chunk
                logger.info(f"[TTS-OVERLAP] Playout completed for sentence: '{sentence}'")

        finally:
            # Clean up all background tasks
            producer_task.cancel()
            for task in list(active_tasks):
                task.cancel()


def _find_all(s: str, char: str):
    """Yield all indices where char appears in s."""
    idx = 0
    while True:
        idx = s.find(char, idx)
        if idx == -1:
            break
        yield idx
        idx += 1
