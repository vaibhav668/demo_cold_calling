import uuid
import json
import asyncio
import time
import re
import contextlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
HAS_DB = True
select = None
AsyncSession = None
get_db_session = None
VoiceProfile = None
VoiceProfileRepository = None
Campaign = None
Customer = None

try:
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.db.session import get_db_session
    from app.voice_demo.models.voice_profile import VoiceProfile
    from app.voice_demo.repositories.voice_profile import VoiceProfileRepository
    from app.models.campaign import Campaign
    from app.models.customer import Customer
except ImportError:
    HAS_DB = False

if HAS_DB:
    db_dependency = Depends(get_db_session)
else:
    async def get_db_session_dummy():
        yield None
    db_dependency = Depends(get_db_session_dummy)

from app.voice_demo.schemas.voice_agent import VoiceProfileOut, SessionSetupIn, SessionSetupOut, SummaryOut
from app.services.conversation_engine import ConversationEngine
from app.services.tts_service import VoiceService, get_voice_service
from app.services.stt_service import SpeechService
from app.services.vad_service import EndOfSpeechDetector
from app.services.session_manager import SessionManager
from app.services.call_state_machine import CallStateMachine, CallState
from app.core.logging import logger
from app.core.telemetry import STARTUP_METRICS

router = APIRouter()

_demo_sessions: Dict[str, Dict[str, Any]] = {}
_STOP_SENTINEL = object()
_pregen_tasks = set()
# Process-level greeting cache keyed by (voice_name, language, industry)
# Populated during startup or first session; reused for subsequent sessions with same config
_greeting_cache: Dict[tuple, list] = {}

def get_greeting_text(industry: str, lang: str, agent_name: str) -> str:
    """Return static greeting text according to selected industry & language."""
    industry = (industry or "").strip().lower()
    lang = (lang or "").strip().lower()
    if industry == "hospital":
        if lang == "hindi":
            return f"नमस्ते! मैं सिटीकेयर हॉस्पिटल से {agent_name} बात कर रही हूँ। क्या मैं आपका नाम जान सकती हूँ?"
        elif lang == "telugu":
            return f"నమస్కారం! నేను సిటీకేర్ హాస్పిటల్ నుండి {agent_name} మాట్లాడుతున్నాను. మీ పేరు తెలుసుకోవచ్చా?"
        else:
            return f"Hi, this is {agent_name} from CityCare Hospital. May I know whom I'm speaking with?"
    else: # real_estate
        if lang == "hindi":
            return f"नमस्ते! मैं स्काईलाइन डेवलपर्स से {agent_name} बात कर रही हूँ। क्या मैं आपका नाम जान सकती हूँ?"
        elif lang == "telugu":
            return f"నమస్కారం! నేను స్కైలైన్ డెవలపర్స్ నుండి {agent_name} మాట్లాడుతున్నాను. మీ పేరు తెలుసుకోవచ్చా?"
        else:
            return f"Hi, this is {agent_name} from Skyline Developers. May I know whom I'm speaking with?"

async def pregenerate_greeting(session_id: str, industry: str, language: str, agent_name: str):
    try:
        from app.services.tts_service import get_voice_service
        language_code = {"English": "en", "Hindi": "hi", "Telugu": "te"}.get(language, "en")
        text = get_greeting_text(industry, language, agent_name)
        cache_key = (agent_name.lower(), language, industry)

        # Check process-level greeting cache first
        if cache_key in _greeting_cache:
            if session_id in _demo_sessions:
                _demo_sessions[session_id]["pregenerated_greeting"] = _greeting_cache[cache_key]
                logger.info(f"[Kokoro-PreGen] Served cached greeting for session {session_id} ({len(_greeting_cache[cache_key])} frames, key={cache_key})")
            return

        tts = get_voice_service()
        chunks = []
        async for chunk in tts.stream_speech(
            text,
            language=language_code,
            voice_config={"persona_name": agent_name}
        ):
            chunks.append(chunk)

        if chunks:
            # Store in the process-level cache for reuse across sessions
            _greeting_cache[cache_key] = chunks
            logger.info(f"[Kokoro-PreGen] Cached greeting for key={cache_key} ({len(chunks)} frames)")

        if session_id in _demo_sessions:
            _demo_sessions[session_id]["pregenerated_greeting"] = chunks
            logger.info(f"[Kokoro-PreGen] Pregenerated greeting for session {session_id} ({len(chunks)} frames)")
    except Exception as e:
        logger.error(f"[Kokoro-PreGen] Failed to pregenerate greeting for {session_id}: {e}")


def pcm16_to_ulaw(pcm_bytes: bytes) -> bytes:
    """Convert raw 16-bit linear PCM bytes to G.711 mu-law via audioop (C-level)."""
    import audioop
    try:
        return audioop.lin2ulaw(pcm_bytes, 2)
    except Exception as e:
        logger.error(f"[AUDIO-TRANSCODE] Failed to convert PCM to mu-law: {e}")
        return b""


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


@router.get("/voices", response_model=List[VoiceProfileOut])
async def get_voices(db: Any = db_dependency):
    """Fetch all active voice profiles available for the browser demo."""
    if HAS_DB and db is not None:
        repo = VoiceProfileRepository(db)
        return await repo.get_active()
    else:
        return VOICE_PROFILES


@router.get("/industries")
async def get_industries():
    """Retrieve supported demo industries and their structural setups."""
    return [
        {
            "id": "hospital",
            "name": "Hospital Receptionist",
            "description": "Engage with Sarah at Mercy Hospital. Confirm details of your appointment, ask about visiting hours, parking fees, or cancellation terms."
        },
        {
            "id": "real_estate",
            "name": "Real Estate Consultant",
            "description": "Speak with James at Premium Realty to qualify for Orchard Heights luxury apartments, ask about amenities, pricing, or book a site visit."
        }
    ]


@router.post("/sessions", response_model=SessionSetupOut)
async def create_session(setup: SessionSetupIn, db: Any = db_dependency):
    """
    Initialize a browser voice demo session.
    Automatically resolves corresponding Campaign & Customer, and performs
    voice-to-language adaptation if needed.
    """
    if HAS_DB and db is not None:
        repo = VoiceProfileRepository(db)
        selected_voice = await repo.get(setup.voice_profile_id)
        if not selected_voice or selected_voice.status != "active":
            raise HTTPException(status_code=404, detail="Selected voice profile not found or inactive.")

        resolved_voice = selected_voice
        supported_langs = [l.strip() for l in selected_voice.supported_languages.split(",")]

        if setup.language not in supported_langs:
            logger.info(f"[VOICE ADAPT] Selected voice {selected_voice.name} does not support {setup.language}. Finding compatible voice...")
            all_voices = await repo.get_active()
            compatible_voice = None

            for v in all_voices:
                v_langs = [l.strip() for l in v.supported_languages.split(",")]
                if v.gender == selected_voice.gender and setup.language in v_langs:
                    compatible_voice = v
                    break

            if not compatible_voice:
                for v in all_voices:
                    v_langs = [l.strip() for l in v.supported_languages.split(",")]
                    if setup.language in v_langs:
                        compatible_voice = v
                        break

            if compatible_voice:
                resolved_voice = compatible_voice
                logger.info(f"[VOICE ADAPT] Switched session voice to {resolved_voice.name} ({resolved_voice.gender})")
            else:
                logger.warning(f"[VOICE ADAPT] No compatible voice found for language {setup.language}. Keeping {selected_voice.name}.")

        camp_query = select(Campaign).where(Campaign.workflow_type == setup.industry, Campaign.is_active == True)
        camp_res = await db.execute(camp_query)
        campaign = camp_res.scalars().first()

        if not campaign:
            camp_query_fb = select(Campaign).where(Campaign.workflow_type == setup.industry)
            camp_res_fb = await db.execute(camp_query_fb)
            campaign = camp_res_fb.scalars().first()
            if not campaign:
                raise HTTPException(status_code=404, detail=f"No campaign configured for industry '{setup.industry}'.")

        cust_query = select(Customer).where(Customer.phone_number == "+15551234567")
        cust_res = await db.execute(cust_query)
        customer = cust_res.scalars().first()

        custom_vars = {"preferred_language": setup.language}
        if setup.industry == "hospital":
            custom_vars.update({
                "doctor_name": "Dr. Sharma",
                "department": "Orthopedics",
                "appointment_date": "tomorrow",
                "appointment_time": "11:00 AM",
                "hospital_name": "CityCare Hospital",
                "purpose": "Routine Consultation"
            })
        else:
            custom_vars.update({
                "property_name": "3 BHK Apartment",
                "property_interest": "3 BHK Apartment",
                "location": "Hyderabad",
                "price": "80 Lakhs",
                "budget": "80 Lakhs",
                "builder": "Skyline Developers"
            })

        if not customer:
            logger.info("[SESSION] Customer Vaibhav not found by phone number. Creating customer...")
            customer = Customer(
                id=uuid.uuid4(),
                first_name="Vaibhav",
                last_name="",
                phone_number="+15551234567",
                email="vaibhav.demo@example.com",
                custom_variables=custom_vars,
                is_active=True
            )
            db.add(customer)
        else:
            logger.info("[SESSION] Customer Vaibhav found by phone number. Updating variables and name...")
            customer.first_name = "Vaibhav"
            customer.last_name = ""
            customer.email = "vaibhav.demo@example.com"
            customer.custom_variables = custom_vars
            customer.is_active = True

        await db.flush()
        await db.commit()

        session_id = str(uuid.uuid4())
        voice_config_dict = json.loads(resolved_voice.voice_configuration or "{}")

        # Start pre-generating greeting in the background
        pg_task = asyncio.create_task(pregenerate_greeting(session_id, setup.industry, setup.language, resolved_voice.name))
        _pregen_tasks.add(pg_task)
        pg_task.add_done_callback(_pregen_tasks.discard)

        sm_manager = SessionManager()
        await sm_manager.update_session_metadata(session_id, {
            "session_id": session_id,
            "campaign_id": str(campaign.id),
            "customer_id": str(customer.id),
            "language": setup.language,
            "agent_name": resolved_voice.name,
            "voice_config": voice_config_dict
        })

        _demo_sessions[session_id] = {
            "session_id": session_id,
            "campaign_id": campaign.id,
            "customer_id": customer.id,
            "voice_profile": resolved_voice,
            "voice_config": voice_config_dict,
            "language": setup.language,
            "industry": setup.industry,
            "created_at": datetime.now(timezone.utc),
            "start_time": None,
            "end_time": None,
            "pregenerate_task": pg_task,
        }

        # Build schema representation of Resolved Voice
        resolved_voice_out = VoiceProfileOut(
            id=resolved_voice.id,
            name=resolved_voice.name,
            description=resolved_voice.description,
            avatar=resolved_voice.avatar,
            gender=resolved_voice.gender,
            supported_languages=resolved_voice.supported_languages,
            voice_provider=resolved_voice.voice_provider,
            preview_audio=resolved_voice.preview_audio,
            status=resolved_voice.status,
            created_at=resolved_voice.created_at,
            updated_at=resolved_voice.updated_at
        )

        return SessionSetupOut(
            session_id=session_id,
            campaign_id=campaign.id,
            customer_id=customer.id,
            voice_profile=resolved_voice_out
        )

    # Database-less / Offline mode
    else:
        voice_name = setup.voice_name or "Sophia"
        selected_voice = next((v for v in VOICE_PROFILES if v["name"].lower() == voice_name.lower()), VOICE_PROFILES[0])

        session_id = str(uuid.uuid4())
        campaign_id = uuid.uuid5(uuid.NAMESPACE_DNS, setup.industry)
        customer_id = uuid.uuid5(uuid.NAMESPACE_DNS, "Vaibhav")

        # Start pre-generating greeting in the background
        pg_task = asyncio.create_task(pregenerate_greeting(session_id, setup.industry, setup.language, selected_voice["name"]))
        _pregen_tasks.add(pg_task)
        pg_task.add_done_callback(_pregen_tasks.discard)

        # Store voice_config with persona_name so Kokoro can resolve the correct voice
        sm_manager = SessionManager()
        await sm_manager.update_session_metadata(session_id, {
            "session_id": session_id,
            "campaign_id": str(campaign_id),
            "customer_id": str(customer_id),
            "language": setup.language,
            "agent_name": selected_voice["name"],
            "voice_config": {"persona_name": selected_voice["name"]}
        })

        _demo_sessions[session_id] = {
            "session_id": session_id,
            "campaign_id": campaign_id,
            "customer_id": customer_id,
            "voice_profile": selected_voice,
            "voice_config": {"persona_name": selected_voice["name"]},
            "language": setup.language,
            "industry": setup.industry,
            "created_at": datetime.now(timezone.utc),
            "start_time": None,
            "end_time": None,
            "pregenerate_task": pg_task,
        }

        voice_out = VoiceProfileOut(
            name=selected_voice["name"],
            description=selected_voice.get("description"),
            avatar=selected_voice.get("avatar"),
            gender=selected_voice["gender"],
            supported_languages=selected_voice["supported_languages"],
            preview_audio=selected_voice.get("preview_audio"),
            status=selected_voice["status"]
        )

        return SessionSetupOut(
            session_id=session_id,
            campaign_id=campaign_id,
            customer_id=customer_id,
            voice_profile=voice_out
        )


@router.api_route("/summary/{session_id}", methods=["GET", "POST"], response_model=SummaryOut)
async def get_session_summary(session_id: str):
    """Return session metadata and status information."""
    meta = _demo_sessions.get(session_id)
    if not meta:
        return SummaryOut(
            summary="Session metadata lost or process restarted.",
            intent="None",
            sentiment="Neutral",
            duration_seconds=0,
            extracted_information={},
            lead_qualification="Not Applicable",
            appointment_status="None",
            knowledge_retrieved=[],
            recommended_next_action="Please restart the conversation.",
            transcript=[],
            language="English",
            voice_used="Sophia",
            industry="hospital",
            lead_score=0,
            site_visit_status="None",
            extracted_variables={},
            session_id=session_id,
            current_state="UNKNOWN (Session lost)",
            failure_reason="Session ID not found in memory",
            error_stack=None
        )

    start = meta.get("start_time")
    end = meta.get("end_time") or time.time()
    duration = int(end - start) if start else 0

    voice_used = "Sophia"
    if meta.get("voice_profile"):
        vp = meta.get("voice_profile")
        if isinstance(vp, dict):
            voice_used = vp.get("name", "Sophia")
        else:
            voice_used = vp.name
    language = meta.get("language", "English")
    industry = meta.get("industry", "hospital")
    failure_reason = meta.get("failure_reason")
    error_stack = meta.get("error_stack")
    current_state = meta.get("current_state", "COMPLETED" if not failure_reason else "FAILED")

    return SummaryOut(
        summary="Call session complete.",
        intent="None",
        sentiment="Neutral",
        duration_seconds=duration,
        extracted_information={},
        lead_qualification="Not Applicable",
        appointment_status="None",
        knowledge_retrieved=[],
        recommended_next_action="None",
        transcript=[],
        language=language,
        voice_used=voice_used,
        industry=industry,
        lead_score=0,
        site_visit_status="None",
        extracted_variables={},
        session_id=session_id,
        current_state=current_state,
        failure_reason=failure_reason,
        error_stack=error_stack
    )


async def _safe_cancel_task(task: Optional[asyncio.Task], timeout: float = 2.0) -> None:
    if task is None or task.done():
        return
    task.cancel()


@router.websocket("/stream/{session_id}")
async def voice_agent_websocket(websocket: WebSocket, session_id: str):
    """
    Bidirectional WebSocket for browser voice agent.
    Streams continuous audio, performs streaming STT during user speech,
    runs Progressive LLM → Sentence TTS pipeline, and transmits detailed telemetry.
    """
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

    campaign_id = uuid.UUID(str(meta["campaign_id"])) if meta.get("campaign_id") else uuid.uuid5(uuid.NAMESPACE_DNS, "hospital")
    customer_id = uuid.UUID(str(meta["customer_id"])) if meta.get("customer_id") else uuid.uuid5(uuid.NAMESPACE_DNS, "Vaibhav")
    language = meta["language"]
    voice_config = meta.get("voice_config", {})
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

    # VAD timing tracker
    vad_timings = []
    greeting_completed = False

    async def _send_state_change(new_state: CallState) -> None:
        """Transition the call state machine and notify the browser of the state change."""
        nonlocal greeting_completed
        await sm.transition(new_state)
        if new_state == CallState.WAITING_FOR_CUSTOMER:
            greeting_completed = True
        try:
            await websocket.send_json({
                "event": "state_change",
                "state": new_state.name
            })
        except Exception:
            pass

    async def _barge_in() -> None:
        """Stop current AI speech immediately and transition to customer speaking."""
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

    async def _fire_pipeline(user_text: str, user_speech_end_t: float = 0.0) -> None:
        """Launch a new progressive streaming pipeline task with the current nonce."""
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
                customer_id=customer_id,
                audio_queue=audio_queue,
                cancel_event=cancel_event,
                sm=sm,
                llm_lock=llm_lock,
                voice_config=voice_config,
                language_code=language_code,
                websocket=websocket,
                session_meta=meta,
                state_callback=_send_state_change,
                nonce=my_nonce,
                get_nonce=lambda: _pipeline_nonce,
                user_speech_end_t=user_speech_end_t,
                vad_timings=vad_timings,
            )
        )

    # ── Audio send loop ──────────────────────────────────────────────────────
    async def _send_loop() -> None:
        chunks_sent = 0
        try:
            while not sm.is_terminal():
                item = await audio_queue.get()

                if item is _STOP_SENTINEL:
                    drained = 0
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                            drained += 1
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
                    chunks_sent += 1
                except Exception as e:
                    logger.error(f"[WS-SEND] Connection lost during audio stream: {e}")
                    break

        except Exception as e:
            logger.error(f"[WS-SEND] Send loop error: {e}")

    send_task = asyncio.create_task(_send_loop())

    # ── Main receive loop ────────────────────────────────────────────────────
    try:
        logger.info(f"[DEMO-WS] Firing sub-second greeting pipeline for session {session_id}")
        await _fire_pipeline("[CALL_START]")

        while not sm.is_terminal():
            data = await websocket.receive()

            if data.get("type") == "websocket.disconnect":
                logger.info(f"[DEMO-WS] Browser disconnected for session {session_id}")
                break

            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    event = msg.get("event")
                    if event == "ping":
                        await websocket.send_json({"event": "pong"})
                    elif event == "stop":
                        logger.info(f"[DEMO-WS] Stop event received for session {session_id}")
                        break
                except Exception:
                    pass

            elif "bytes" in data:
                binary_data = data["bytes"]
                mu_law_audio = pcm16_to_ulaw(binary_data)

                # Measure VAD latency
                v_start = time.perf_counter()

                # VAD during AI speech: barge-in detection
                if sm.is_ai_speaking():
                    if not greeting_completed:
                        vad.reset()
                        continue
                    loop_time = loop.time()
                    if loop_time - sm.ai_speech_start_time > 1.2:
                        vad_event = await loop.run_in_executor(None, vad.process_frame, mu_law_audio)
                        v_elapsed = (time.perf_counter() - v_start) * 1000.0
                        vad_timings.append(v_elapsed)

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
                v_elapsed = (time.perf_counter() - v_start) * 1000.0
                vad_timings.append(v_elapsed)

                if sm.state == CallState.CUSTOMER_SPEAKING:
                    utterance_buffer.extend(mu_law_audio)

                    # Streaming STT: periodically run intermediate transcription every 8000 bytes (~1.0s audio)
                    if len(utterance_buffer) - last_intermediate_stt_len >= 8000:
                        last_intermediate_stt_len = len(utterance_buffer)

                        async def _run_intermediate_stt(audio_snapshot: bytes):
                            try:
                                inter_transcript = await stt.transcribe_utterance(audio_snapshot, language=language_code)
                                if inter_transcript:
                                    await websocket.send_json({
                                        "event": "transcript",
                                        "sender": "user",
                                        "text": inter_transcript,
                                        "intermediate": True
                                    })
                            except Exception:
                                pass

                        # Launch intermediate STT in background without blocking
                        if intermediate_stt_task is None or intermediate_stt_task.done():
                            intermediate_stt_task = asyncio.create_task(_run_intermediate_stt(bytes(utterance_buffer)))

                if vad_event == "speech_start":
                    if sm.is_waiting():
                        logger.info(f"[DEMO-WS] Speech start detected for session {session_id}")
                        utterance_buffer.clear()
                        last_intermediate_stt_len = 0
                        vad.reset()
                        vad.provider._in_speech = True
                        if hasattr(vad.provider, '_speech_confirmed'):
                            vad.provider._speech_confirmed = True
                        await _send_state_change(CallState.CUSTOMER_SPEAKING)

                elif vad_event == "speech_end":
                    if sm.state == CallState.CUSTOMER_SPEAKING:
                        user_speech_end_t = time.perf_counter()
                        utterance_bytes = bytes(utterance_buffer)
                        utterance_buffer.clear()
                        last_intermediate_stt_len = 0
                        vad.reset()

                        await _safe_cancel_task(intermediate_stt_task)
                        intermediate_stt_task = None

                        # Check minimum duration threshold (400ms = 3200 bytes for 8kHz mu-law)
                        duration_ms = len(utterance_bytes) / 8.0
                        if duration_ms < 400.0:
                            logger.info(f"[DEMO-WS] Utterance too short ({duration_ms:.0f}ms < 400ms). Ignoring noise/click and returning to WAITING_FOR_CUSTOMER.")
                            await _send_state_change(CallState.WAITING_FOR_CUSTOMER)
                            continue

                        await _send_state_change(CallState.TRANSCRIBING)

                        async def _transcribe_and_run(audio: bytes, speech_end_t: float) -> None:
                            try:
                                # Determine if this is the first user response (to optimize name recognition)
                                sm_manager = SessionManager()
                                messages = await sm_manager.get_message_history(session_id)
                                user_msgs = [m for m in messages if m["role"] == "user"]
                                is_first_turn = (len(user_msgs) == 0)

                                prompt_hint = None
                                if is_first_turn:
                                    prompt_hint = "The user is introducing themselves by stating their name, for example: I'm Akash, My name is Arjun, Sophia, David, Maya."

                                _stt_start = time.perf_counter()
                                transcript = await stt.transcribe_utterance(audio, language=language_code, prompt=prompt_hint)
                                stt_latency_ms = (time.perf_counter() - _stt_start) * 1000.0

                                if is_first_turn and transcript:
                                    from app.services.conversation_engine import normalize_name_transcript
                                    transcript = normalize_name_transcript(transcript)

                                if not transcript:
                                    logger.info(f"[DEMO-WS] Empty transcript. Playing recovery prompt.")
                                    await _send_state_change(CallState.WAITING_FOR_CUSTOMER)
                                    await _fire_pipeline("[RECOVERY_SAY:Sorry, I didn't catch that. Could you repeat that once?]")
                                    return

                                logger.info(f"[METRICS] STT Latency: {stt_latency_ms:.1f}ms | Transcript: '{transcript}'")

                                # Send final user transcript to browser
                                try:
                                    await websocket.send_json({
                                        "event": "transcript",
                                        "sender": "user",
                                        "text": transcript,
                                        "intermediate": False
                                    })
                                except Exception:
                                    pass

                                await _fire_pipeline(transcript, user_speech_end_t=speech_end_t)
                            except Exception as e:
                                import traceback
                                stack = traceback.format_exc()
                                logger.error(f"[DEMO-WS] Producer task failed with error: {e}\n{stack}")
                                try:
                                    await _send_state_change(CallState.WAITING_FOR_CUSTOMER)
                                    await _fire_pipeline("[RECOVERY_SAY:Sorry, I encountered an issue processing that. Could you say it again?]")
                                except Exception:
                                    pass

                        await _safe_cancel_task(pipeline_task)
                        pipeline_task = asyncio.create_task(_transcribe_and_run(utterance_bytes, user_speech_end_t))

    except WebSocketDisconnect as e:
        logger.info(f"[DEMO-WS] WebSocket disconnect event for session {session_id} (code={e.code}, reason={e.reason or 'None'})")
        if meta:
            meta["failure_reason"] = f"WebSocket disconnected: code={e.code}, reason={e.reason or 'None'}"
            meta["current_state"] = sm.state.name
    except Exception as e:
        import traceback
        stack = traceback.format_exc()
        logger.error(f"[DEMO-WS] WebSocket exception for session {session_id}: {e}", exc_info=True)
        if meta:
            meta["failure_reason"] = f"WebSocket exception: {e}"
            meta["current_state"] = sm.state.name
            meta["error_stack"] = stack
    finally:
        meta["end_time"] = time.time()
        logger.info(f"[DEMO-WS] Cleaning up session {session_id}")

        cancel_event.set()
        await _safe_cancel_task(intermediate_stt_task)
        await _safe_cancel_task(pipeline_task)

        audio_queue.put_nowait(None)
        while not audio_queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                audio_queue.get_nowait()

        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(send_task, timeout=2.0)

        with contextlib.suppress(Exception):
            await websocket.close()

        utterance_buffer.clear()
        with contextlib.suppress(Exception):
            await SessionManager().clear_session(session_id)


async def _run_pipeline(
    call_uuid: str,
    user_text: str,
    campaign_id: uuid.UUID,
    customer_id: uuid.UUID,
    audio_queue: asyncio.Queue,
    cancel_event: asyncio.Event,
    sm: CallStateMachine,
    llm_lock: asyncio.Lock,
    voice_config: Optional[dict] = None,
    language_code: Optional[str] = None,
    websocket: Optional[WebSocket] = None,
    session_meta: Optional[dict] = None,
    state_callback=None,
    nonce: int = 0,
    get_nonce=None,
    user_speech_end_t: float = 0.0,
    vad_timings: Optional[List[float]] = None,
) -> None:
    """
    End-to-End Progressive Pipeline:
    Stream LLM Tokens → Sentence Splitter → Progressive TTS Synthesis → Client Audio Queue
    Calculates detailed telemetry metrics and pushes them to the browser.
    """
    _pipeline_start = time.perf_counter()
    agent_name = "Sophia"
    if session_meta:
        agent_name = session_meta.get("agent_name", "Sophia")

    # Ensure persona_name is always in voice_config for correct Kokoro voice routing
    effective_voice_config = dict(voice_config) if voice_config else {}
    if "persona_name" not in effective_voice_config:
        effective_voice_config["persona_name"] = agent_name
    voice_config = effective_voice_config

    def _is_superseded() -> bool:
        return get_nonce is not None and get_nonce() != nonce

    if state_callback:
        await state_callback(CallState.THINKING)

    if _is_superseded():
        return

    should_hangup = False
    should_transfer = False
    llm_first_token_ms = 0.0
    tts_first_byte_ms = 0.0
    total_round_trip_ms = 0.0
    chunks_count = 0
    full_agent_response = []

    t_llm_start = time.time()
    served_pregen = False

    if user_text.upper() == "[CALL_START]":
        meta_info = _demo_sessions.get(call_uuid)
        if meta_info:
            task = meta_info.get("pregenerate_task")
            if task and not task.done():
                logger.info(f"[Kokoro-PreGen] Background greeting pre-generation task not completed. Bypassing wait to avoid blocking connection.")
                task.cancel()
            
            if meta_info.get("pregenerated_greeting"):
                served_pregen = True
                logger.info(f"[Kokoro-PreGen] Serving pregenerated greeting for session {call_uuid}")
                if state_callback:
                    await state_callback(CallState.GENERATING_RESPONSE)
                for chunk in meta_info["pregenerated_greeting"]:
                    await audio_queue.put(chunk)
                first_audio_ms = (time.time() - t_llm_start) * 1000.0
                logger.info(f"[TURN] FIRST-AUDIO-BYTE (PRE-GENERATED) | session={call_uuid} latency={first_audio_ms:.0f}ms")
                if state_callback:
                    await state_callback(CallState.AI_SPEAKING)
                    sm.ai_speech_start_time = asyncio.get_event_loop().time()
                
                # Update dialog history and session state
                try:
                    sm_manager = SessionManager()
                    greeting_text = get_greeting_text(
                        meta_info.get("industry", "hospital"),
                        meta_info.get("language", "English"),
                        agent_name
                    )
                    await sm_manager.append_message(call_uuid, {"role": "assistant", "content": greeting_text})
                    await sm_manager.update_session_state(call_uuid, "WAIT_FOR_NAME")
                except Exception as e:
                    logger.error(f"[DEMO-PIPELINE] Failed to update greeting state: {e}")
                if state_callback:
                    await state_callback(CallState.WAITING_FOR_CUSTOMER)
                return

    _llm_start = time.perf_counter()

    async with llm_lock:
        if _is_superseded():
            return

        try:
            db = None
            if HAS_DB and get_db_session is not None:
                try:
                    async for db_session in get_db_session():
                        db = db_session
                        break
                except Exception:
                    pass

            engine = ConversationEngine(db)
            tts = get_voice_service()

            # Stream LLM token generator
            if user_text.upper() == "[CALL_START]" and not served_pregen:
                meta_info = _demo_sessions.get(call_uuid)
                greeting_text = get_greeting_text(
                    meta_info.get("industry", "hospital") if meta_info else "hospital",
                    meta_info.get("language", "English") if meta_info else "English",
                    agent_name
                )
                async def _dummy_stream():
                    yield greeting_text, False, False
                token_stream = _dummy_stream()
            elif user_text.startswith("[RECOVERY_SAY:"):
                phrase = user_text[14:-1]
                async def _dummy_stream():
                    yield phrase, False, False
                token_stream = _dummy_stream()
            else:
                if HAS_DB and db is not None:
                    token_stream = engine.process_turn_stream(
                        call_id=call_uuid,
                        campaign_id=campaign_id,
                        customer_id=customer_id,
                        user_text=user_text
                    )
                else:
                    industry = session_meta.get("industry", "hospital") if session_meta else "hospital"
                    language = session_meta.get("language", "English") if session_meta else "English"
                    token_stream = engine.process_turn_stream(
                        call_id=call_uuid,
                        campaign_id=campaign_id,
                        industry=industry,
                        language=language,
                        agent_name=agent_name,
                        user_text=user_text
                    )

            # Generator yielding raw text chunks for TTS
            async def _text_chunk_extractor():
                nonlocal llm_first_token_ms, should_hangup, should_transfer
                async for chunk, h, tr in token_stream:
                    if _is_superseded() or cancel_event.is_set():
                        break
                    if h:
                        should_hangup = True
                    if tr:
                        should_transfer = True
                    if chunk:
                        if llm_first_token_ms == 0.0:
                            llm_first_token_ms = (time.perf_counter() - _llm_start) * 1000.0
                        full_agent_response.append(chunk)
                        yield chunk

            # Pass text generator into progressive sentence-level TTS streamer
            _tts_start = time.perf_counter()
            audio_stream = tts.stream_text_stream_progressive(
                _text_chunk_extractor(),
                cancel_event=cancel_event,
                language=language_code,
                voice_config=voice_config
            )

            # Transition state to GENERATING_RESPONSE / AI_SPEAKING as soon as audio starts
            if state_callback:
                await state_callback(CallState.GENERATING_RESPONSE)

            first_chunk_sent = False

            async for audio_chunk in audio_stream:
                if _is_superseded() or cancel_event.is_set():
                    break

                if not first_chunk_sent:
                    first_chunk_sent = True
                    tts_first_byte_ms = (time.perf_counter() - _tts_start) * 1000.0
                    if user_speech_end_t > 0.0:
                        total_round_trip_ms = (time.perf_counter() - user_speech_end_t) * 1000.0
                    else:
                        total_round_trip_ms = (time.perf_counter() - _pipeline_start) * 1000.0

                    if state_callback:
                        await state_callback(CallState.AI_SPEAKING)

                    # Transmit real-time telemetry metrics to browser
                    avg_vad_ms = round(sum(vad_timings) / len(vad_timings), 2) if vad_timings else 0.0
                    try:
                        if websocket:
                            await websocket.send_json({
                                "event": "metrics",
                                "metrics": {
                                    "llm_latency_ms": round(llm_first_token_ms, 1),
                                    "tts_first_byte_ms": round(tts_first_byte_ms, 1),
                                    "total_round_trip_ms": round(total_round_trip_ms, 1),
                                    "vad_latency_ms": avg_vad_ms,
                                }
                            })
                    except Exception:
                        pass

                    logger.info(
                        f"[TELEMETRY] Round-Trip={total_round_trip_ms:.1f}ms | "
                        f"LLM TTFT={llm_first_token_ms:.1f}ms | TTS TTFB={tts_first_byte_ms:.1f}ms | VAD avg={avg_vad_ms}ms"
                    )

                await audio_queue.put(audio_chunk)
                chunks_count += 1

        except asyncio.CancelledError:
            raise
        except Exception as e:
            import traceback
            stack = traceback.format_exc()
            logger.error(f"[DEMO-PIPELINE] Pipeline error: {e}\n{stack}")

    _pipeline_total = (time.perf_counter() - _pipeline_start) * 1000.0
    full_text_str = "".join(full_agent_response).strip()

    if full_text_str and websocket:
        try:
            await websocket.send_json({
                "event": "transcript",
                "sender": "agent",
                "text": full_text_str,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        except Exception:
            pass

    logger.info(f"[METRICS] Pipeline Complete: total={_pipeline_total:.1f}ms | chunks={chunks_count}")

    if not _is_superseded() and not cancel_event.is_set():
        if user_text.upper() == "[CALL_START]":
            # For fallback real-time greeting, update dialogue history and session state
            try:
                meta_info = _demo_sessions.get(call_uuid)
                sm_manager = SessionManager()
                greeting_text = get_greeting_text(
                    meta_info.get("industry", "hospital") if meta_info else "hospital",
                    meta_info.get("language", "English") if meta_info else "English",
                    agent_name
                )
                await sm_manager.append_message(call_uuid, {"role": "assistant", "content": greeting_text})
                await sm_manager.update_session_state(call_uuid, "WAIT_FOR_NAME")
            except Exception as e:
                logger.error(f"[DEMO-PIPELINE] Failed to update greeting state on fallback: {e}")

        if should_hangup:
            if state_callback:
                await state_callback(CallState.CALL_COMPLETED)
        else:
            if state_callback:
                await state_callback(CallState.WAITING_FOR_CUSTOMER)
