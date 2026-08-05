import uuid
import json
import asyncio
import time
import contextlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status

from app.voice_demo.schemas.voice_agent import VoiceProfileOut, SessionSetupIn, SessionSetupOut
from app.services.conversation_engine import ConversationEngine
from app.services.tts_service import VoiceService
from app.services.stt_service import SpeechService
from app.services.vad_service import EndOfSpeechDetector
from app.services.session_manager import SessionManager
from app.services.call_state_machine import CallStateMachine, CallState
from app.core.logging import logger
from app.core.telemetry import STARTUP_METRICS

router = APIRouter()

# In-memory dictionary to track active browser sessions
_demo_sessions: Dict[str, Dict[str, Any]] = {}
_STOP_SENTINEL = object()

VOICE_PROFILES = [
    {
        "name": "Sophia",
        "description": "Professional Female",
        "avatar": "/static/images/avatars/sophia.png",
        "gender": "Female",
        "supported_languages": "English,Hindi,Telugu",
        "preview_audio": "/static/audio/previews/sophia.mp3",
        "status": "active"
    },
    {
        "name": "Maya",
        "description": "Friendly Female",
        "avatar": "/static/images/avatars/maya.png",
        "gender": "Female",
        "supported_languages": "English,Hindi",
        "preview_audio": "/static/audio/previews/maya.mp3",
        "status": "active"
    },
    {
        "name": "Ananya",
        "description": "Customer Support Female",
        "avatar": "/static/images/avatars/ananya.png",
        "gender": "Female",
        "supported_languages": "English,Telugu",
        "preview_audio": "/static/audio/previews/ananya.mp3",
        "status": "active"
    },
    {
        "name": "Arjun",
        "description": "Professional Male",
        "avatar": "/static/images/avatars/arjun.png",
        "gender": "Male",
        "supported_languages": "English,Hindi,Telugu",
        "preview_audio": "/static/audio/previews/arjun.mp3",
        "status": "active"
    },
    {
        "name": "David",
        "description": "Sales Consultant Male",
        "avatar": "/static/images/avatars/david.png",
        "gender": "Male",
        "supported_languages": "English",
        "preview_audio": "/static/audio/previews/david.mp3",
        "status": "active"
    }
]

def pcm16_to_ulaw(pcm_bytes: bytes) -> bytes:
    import audioop
    try:
        return audioop.lin2ulaw(pcm_bytes, 2)
    except Exception as e:
        logger.error(f"[AUDIO-TRANSCODE] Failed to convert PCM to mu-law: {e}")
        return b""

@router.get("/voices", response_model=List[VoiceProfileOut])
async def get_voices():
    return VOICE_PROFILES

@router.get("/preview")
async def get_preview_audio(voice: str, lang: str = "English"):
    from fastapi.responses import StreamingResponse
    import edge_tts
    import io

    language_code = {"English": "en", "Hindi": "hi", "Telugu": "te"}.get(lang, "en")
    # Resolve voice name using mapping function
    from app.services.speech.tts.edge_tts_provider import _resolve_voice
    resolved_voice = _resolve_voice({"name": voice}, language_code)

    if language_code == "hi":
        text = f"नमस्ते, मेरा नाम {voice} है। आपसे बात करके खुशी होगी।"
    elif language_code == "te":
        text = f"నమస్కారం, నా పేరు {voice}. మీతో మాట్లాడటం చాలా సంతోషంగా ఉంది."
    else:
        text = f"Hello, my name is {voice}. I am looking forward to speaking with you."

    try:
        communicate = edge_tts.Communicate(text, resolved_voice)
        mp3_buffer = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.extend(chunk["data"])

        return StreamingResponse(io.BytesIO(mp3_buffer), media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"Failed to generate preview audio: {e}")
        raise HTTPException(status_code=500, detail="Preview generation failed.")


@router.post("/sessions", response_model=SessionSetupOut)
async def create_session(setup: SessionSetupIn):
    selected_voice = next((v for v in VOICE_PROFILES if v["name"].lower() == setup.voice_name.lower()), None)
    if not selected_voice:
        raise HTTPException(status_code=404, detail="Selected voice profile not found.")

    session_id = str(uuid.uuid4())
    # Pre-calculate deterministic campaign_id based on selected industry
    campaign_id = uuid.uuid5(uuid.NAMESPACE_DNS, setup.industry)

    sm_manager = SessionManager()
    await sm_manager.update_session_metadata(session_id, {
        "session_id": session_id,
        "campaign_id": str(campaign_id),
        "language": setup.language,
        "agent_name": selected_voice["name"],
        "industry": setup.industry
    })

    _demo_sessions[session_id] = {
        "session_id": session_id,
        "campaign_id": campaign_id,
        "voice_profile": selected_voice,
        "language": setup.language,
        "industry": setup.industry,
        "created_at": datetime.now(timezone.utc),
        "start_time": None,
        "end_time": None,
    }

    return SessionSetupOut(
        session_id=session_id,
        campaign_id=campaign_id,
        voice_profile=selected_voice
    )

async def _safe_cancel_task(task: Optional[asyncio.Task], timeout: float = 2.0) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError, Exception):
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

@router.websocket("/stream/{session_id}")
async def voice_agent_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"[DEMO-WS] Connected session: {session_id}")

    meta = _demo_sessions.get(session_id)
    if not meta:
        logger.error(f"[DEMO-WS] Session meta not found for {session_id}. Terminating.")
        await websocket.close(code=1008)
        return

    meta["start_time"] = time.time()

    # Transmit startup telemetry to client upon connection
    try:
        await websocket.send_json({
            "event": "startup_metrics",
            "metrics": STARTUP_METRICS
        })
    except Exception:
        pass

    campaign_id = meta["campaign_id"]
    language = meta["language"]
    voice_profile = meta["voice_profile"]
    industry = meta["industry"]
    language_code = {"English": "en", "Hindi": "hi", "Telugu": "te"}.get(language, "en")

    sm = CallStateMachine(session_id)
    audio_queue: asyncio.Queue = asyncio.Queue()
    llm_lock = asyncio.Lock()
    cancel_event = asyncio.Event()

    utterance_buffer = bytearray()
    last_intermediate_stt_len = 0
    intermediate_stt_task: Optional[asyncio.Task] = None

    vad = EndOfSpeechDetector()
    stt = SpeechService()

    pipeline_task: Optional[asyncio.Task] = None
    _pipeline_nonce = 0
    loop = asyncio.get_event_loop()

    async def _send_state_change(new_state: CallState) -> None:
        await sm.transition(new_state)
        try:
            await websocket.send_json({
                "event": "state_change",
                "state": new_state.name
            })
        except Exception:
            pass

    async def _barge_in() -> None:
        nonlocal pipeline_task, _pipeline_nonce, cancel_event, intermediate_stt_task
        logger.info(f"[BARGE-IN] Customer interrupted AI speech for session {session_id}")

        _pipeline_nonce += 1
        cancel_event.set()
        audio_queue.put_nowait(_STOP_SENTINEL)
        vad.reset()
        utterance_buffer.clear()

        await _safe_cancel_task(intermediate_stt_task)
        intermediate_stt_task = None

        await _safe_cancel_task(pipeline_task)
        pipeline_task = None

        await _send_state_change(CallState.CUSTOMER_SPEAKING)

    async def _fire_pipeline(user_text: str) -> None:
        nonlocal pipeline_task, _pipeline_nonce, cancel_event

        _pipeline_nonce += 1
        my_nonce = _pipeline_nonce
        cancel_event.clear()

        await _safe_cancel_task(pipeline_task)

        pipeline_task = asyncio.create_task(
            _run_pipeline(
                call_uuid=session_id,
                user_text=user_text,
                campaign_id=campaign_id,
                industry=industry,
                language=language,
                agent_name=voice_profile["name"],
                audio_queue=audio_queue,
                cancel_event=cancel_event,
                sm=sm,
                llm_lock=llm_lock,
                language_code=language_code,
                websocket=websocket,
                state_callback=_send_state_change,
                nonce=my_nonce,
                get_nonce=lambda: _pipeline_nonce,
            )
        )

    # Audio send loop
    async def _send_loop() -> None:
        try:
            while not sm.is_terminal():
                item = await audio_queue.get()

                if item is _STOP_SENTINEL:
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    try:
                        await websocket.send_json({"event": "clear_audio"})
                    except Exception:
                        pass
                    continue

                if item is None:
                    break

                try:
                    await websocket.send_bytes(item)
                except Exception as e:
                    logger.error(f"[WS-SEND] Connection lost: {e}")
                    break

        except Exception as e:
            logger.error(f"[WS-SEND] Send loop error: {e}")

    send_task = asyncio.create_task(_send_loop())

    try:
        # Launch greeting
        await _fire_pipeline("[CALL_START]")

        while not sm.is_terminal():
            data = await websocket.receive()

            if data.get("type") == "websocket.disconnect":
                break

            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    event = msg.get("event")
                    if event == "ping":
                        await websocket.send_json({"event": "pong"})
                    elif event == "stop":
                        break
                except Exception:
                    pass

            elif "bytes" in data:
                binary_data = data["bytes"]
                mu_law_audio = pcm16_to_ulaw(binary_data)

                # VAD during AI speech: barge-in detection
                if sm.is_ai_speaking():
                    loop_time = loop.time()
                    if loop_time - sm.ai_speech_start_time > 1.2:
                        vad_event = await loop.run_in_executor(None, vad.process_frame, mu_law_audio)
                        if vad_event == "speech_start":
                            await _barge_in()
                    else:
                        vad.reset()
                    continue

                if sm.state in (
                    CallState.TRANSCRIBING,
                    CallState.THINKING,
                    CallState.GENERATING_RESPONSE,
                    CallState.CALL_COMPLETED,
                    CallState.ERROR,
                ):
                    continue

                loop_time = loop.time()
                if sm.is_waiting() and (loop_time - sm.waiting_start_time < 0.6):
                    vad.reset()
                    continue

                # Normal VAD processing
                vad_event = await loop.run_in_executor(None, vad.process_frame, mu_law_audio)

                if sm.state == CallState.CUSTOMER_SPEAKING:
                    utterance_buffer.extend(mu_law_audio)

                if vad_event == "speech_start":
                    if sm.is_waiting():
                        utterance_buffer.clear()
                        vad.reset()
                        await _send_state_change(CallState.CUSTOMER_SPEAKING)

                elif vad_event == "speech_end":
                    if sm.state == CallState.CUSTOMER_SPEAKING:
                        await _send_state_change(CallState.TRANSCRIBING)
                        utterance_bytes = bytes(utterance_buffer)
                        utterance_buffer.clear()
                        vad.reset()

                        await _safe_cancel_task(intermediate_stt_task)
                        intermediate_stt_task = None

                        async def _transcribe_and_run(audio: bytes) -> None:
                            transcript = await stt.transcribe_utterance(audio, language=language_code)
                            if not transcript:
                                await _send_state_change(CallState.WAITING_FOR_CUSTOMER)
                                return

                            await _fire_pipeline(transcript)

                        await _safe_cancel_task(pipeline_task)
                        pipeline_task = asyncio.create_task(_transcribe_and_run(utterance_bytes))

    except WebSocketDisconnect:
        logger.info(f"[DEMO-WS] WebSocket disconnect for session {session_id}")
    except Exception as e:
        logger.error(f"[DEMO-WS] WebSocket exception: {e}", exc_info=True)
    finally:
        meta["end_time"] = time.time()
        logger.info(f"[DEMO-WS] Cleaning up session {session_id}")

        cancel_event.set()
        await _safe_cancel_task(intermediate_stt_task)
        await _safe_cancel_task(pipeline_task)

        audio_queue.put_nowait(None)
        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(send_task, timeout=2.0)

        with contextlib.suppress(Exception):
            await websocket.close()

        utterance_buffer.clear()
        # Clean up session COMPLETELY from memory to avoid leaks
        _demo_sessions.pop(session_id, None)
        await SessionManager().clear_session(session_id)

async def _run_pipeline(
    call_uuid: str,
    user_text: str,
    campaign_id: uuid.UUID,
    industry: str,
    language: str,
    agent_name: str,
    audio_queue: asyncio.Queue,
    cancel_event: asyncio.Event,
    sm: CallStateMachine,
    llm_lock: asyncio.Lock,
    language_code: Optional[str] = None,
    websocket: Optional[WebSocket] = None,
    state_callback=None,
    nonce: int = 0,
    get_nonce=None,
) -> None:
    def _is_superseded() -> bool:
        return get_nonce is not None and get_nonce() != nonce

    if state_callback:
        await state_callback(CallState.THINKING)

    if _is_superseded():
        return

    should_hangup = False
    should_transfer = False

    async with llm_lock:
        if _is_superseded():
            return

        try:
            engine = ConversationEngine()
            tts = VoiceService()

            token_stream = engine.process_turn_stream(
                call_id=call_uuid,
                campaign_id=campaign_id,
                industry=industry,
                language=language,
                agent_name=agent_name,
                user_text=user_text
            )

            async def _text_chunk_extractor():
                nonlocal should_hangup, should_transfer
                async for chunk, h, tr in token_stream:
                    if _is_superseded() or cancel_event.is_set():
                        break
                    if h:
                        should_hangup = True
                    if tr:
                        should_transfer = True
                    if chunk:
                        yield chunk

            audio_stream = tts.stream_text_stream_progressive(
                _text_chunk_extractor(),
                cancel_event=cancel_event,
                language=language_code,
                voice_config={"persona_name": agent_name}
            )

            if state_callback:
                await state_callback(CallState.GENERATING_RESPONSE)

            first_chunk_sent = False
            async for audio_chunk in audio_stream:
                if _is_superseded() or cancel_event.is_set():
                    break

                if not first_chunk_sent:
                    first_chunk_sent = True
                    if state_callback:
                        await state_callback(CallState.AI_SPEAKING)
                        # Set actual start timestamp for barge-in checks
                        sm.ai_speech_start_time = asyncio.get_event_loop().time()

                await audio_queue.put(audio_chunk)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[DEMO-PIPELINE] Pipeline error: {e}", exc_info=True)

    if not _is_superseded() and not cancel_event.is_set():
        if should_hangup:
            if state_callback:
                await state_callback(CallState.CALL_COMPLETED)
        else:
            if state_callback:
                await state_callback(CallState.WAITING_FOR_CUSTOMER)
