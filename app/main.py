from contextlib import asynccontextmanager
import time
from typing import Dict, Any
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.core.telemetry import STARTUP_METRICS
from app.voice_demo.controllers.voice_agent import router as voice_agent_router

AppException = None
RequestLoggingMiddleware = None

try:
    from app.core.exceptions import AppException
except ImportError:
    pass

try:
    from app.core.middleware import RequestLoggingMiddleware
except ImportError:
    pass

HAS_PROD_MODULES = True
api_v1_router = None
chroma_manager = None

try:
    from app.api.v1.router import router as api_v1_router
    from app.db.chroma import chroma_manager
except ImportError:
    HAS_PROD_MODULES = False

# Configure logging at startup
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_start = time.perf_counter()

    # Configure Torch optimizations on boot
    try:
        import torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        logger.info("[TORCH] PyTorch optimized for low memory: set_grad_enabled(False), threads=1")
    except ImportError:
        pass

    # Log initial memory usage on boot
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        STARTUP_METRICS["rss_mb"] = round(rss, 2)
        logger.info(f"[MEMORY] Startup initial RSS: {rss:.2f} MB")
    except Exception:
        pass

    # Startup hook: Initialize external clients
    if HAS_PROD_MODULES:
        logger.info("Running database connectivity diagnostics...")
        try:
            from app.db.session import run_db_diagnostics, verify_db_connection
            run_db_diagnostics()
            await verify_db_connection()
        except Exception as e:
            logger.critical(f"Database configuration validation failed! Web server will continue booting. Error: {e}")

    if chroma_manager is not None:
        logger.info("Initializing external service connection pools...")
        chroma_manager.connect()

        try:
            from app.services.rag_service import RAGService
            rag = RAGService()
            await rag.initialize_collection()
        except Exception as e:
            logger.error(f"Failed to auto-initialize RAG collection: {e}")

    # Auto-seed voice profiles if table is empty (idempotent)
    if HAS_PROD_MODULES:
        try:
            from app.db.session import get_session_maker
            from app.voice_demo.models.voice_profile import VoiceProfile
            from sqlalchemy import select, func
            import json as _json
            import uuid as _uuid

            _session_maker = get_session_maker()
            async with _session_maker() as _seed_db:
                count_res = await _seed_db.execute(
                    select(func.count()).select_from(VoiceProfile)
                )
                count = count_res.scalar()
                if count == 0:
                    logger.info("[Startup] voice_profiles table empty — seeding default voices...")
                    _profiles = [
                        VoiceProfile(id=_uuid.UUID("550e8400-e29b-41d4-a716-446655440001"), name="Sophia",  description="Professional Female",    gender="Female", supported_languages="English,Hindi",         voice_provider="melotts", voice_configuration=_json.dumps({"speaker_id": "EN_INDIA", "speed": 0.95}), status="active"),
                        VoiceProfile(id=_uuid.UUID("550e8400-e29b-41d4-a716-446655440002"), name="Maya",    description="Friendly Female",          gender="Female", supported_languages="English,Telugu",        voice_provider="melotts", voice_configuration=_json.dumps({"speaker_id": "EN_INDIA", "speed": 1.0}),  status="active"),
                        VoiceProfile(id=_uuid.UUID("550e8400-e29b-41d4-a716-446655440003"), name="Ananya",  description="Customer Support",         gender="Female", supported_languages="English,Hindi,Telugu",  voice_provider="melotts", voice_configuration=_json.dumps({"speaker_id": "EN_INDIA", "speed": 1.05}), status="active"),
                        VoiceProfile(id=_uuid.UUID("550e8400-e29b-41d4-a716-446655440004"), name="Arjun",   description="Sales Specialist",         gender="Male",   supported_languages="English,Hindi",         voice_provider="melotts", voice_configuration=_json.dumps({"speaker_id": "EN_INDIA", "speed": 1.0}),  status="active"),
                        VoiceProfile(id=_uuid.UUID("550e8400-e29b-41d4-a716-446655440005"), name="David",   description="Enterprise Consultant",    gender="Male",   supported_languages="English",               voice_provider="melotts", voice_configuration=_json.dumps({"speaker_id": "EN_US",    "speed": 0.98}), status="active"),
                    ]
                    _seed_db.add_all(_profiles)
                    await _seed_db.commit()
                    logger.info("[Startup] Voice profiles seeded successfully.")
                else:
                    logger.info(f"[Startup] voice_profiles has {count} profiles — skip seed.")
        except Exception as e:
            logger.error(f"[Startup] Voice profile auto-seed failed (non-fatal): {e}")

    # Eagerly warm up ALL AI pipeline services during server startup (no first-request latency penalty)
    try:
        import asyncio
        w_start = time.perf_counter()

        # 1. Warm up VAD
        v_t0 = time.perf_counter()
        from app.services.speech.vad.silero_provider import SileroVADProvider
        def load_vad():
            v = SileroVADProvider()
            v.process_frame(b"\x00" * 160)
            return v
        await asyncio.get_event_loop().run_in_executor(None, load_vad)
        STARTUP_METRICS["vad_load_ms"] = round((time.perf_counter() - v_t0) * 1000.0, 1)

        # 2. Warm up STT Singleton (Whisper)
        s_t0 = time.perf_counter()
        from app.services.stt_service import SpeechService
        stt_ms = await SpeechService.warmup()
        STARTUP_METRICS["stt_load_ms"] = round(stt_ms, 1)

        # 3. Warm up TTS Provider (including pre-caching all production voice styles)
        t_t0 = time.perf_counter()
        from app.services.tts_service import VoiceService, get_voice_service
        from app.services.speech.tts.kokoro_provider import KokoroProvider
        await VoiceService.warmup()
        STARTUP_METRICS["tts_load_ms"] = round((time.perf_counter() - t_t0) * 1000.0, 1)

        # 4. Warm up LLM Connection Pool (Groq/OpenRouter)
        l_t0 = time.perf_counter()
        try:
            from app.services.llm_service import LLMService
            llm = LLMService()
            await llm.generate_completion([{"role": "user", "content": "hi"}])
        except Exception as llm_err:
            logger.warning(f"[WARMUP] LLM ping non-fatal error: {llm_err}")
        STARTUP_METRICS["llm_warmup_ms"] = round((time.perf_counter() - l_t0) * 1000.0, 1)

        tot_ms = (time.perf_counter() - w_start) * 1000.0
        STARTUP_METRICS["total_warmup_ms"] = round(tot_ms, 1)
        logger.info(
            f"[WARMUP COMPLETE] All AI Subsystems Ready! "
            f"VAD={STARTUP_METRICS['vad_load_ms']}ms | STT={STARTUP_METRICS['stt_load_ms']}ms | "
            f"TTS={STARTUP_METRICS['tts_load_ms']}ms | LLM={STARTUP_METRICS['llm_warmup_ms']}ms"
        )

        # 5. Pre-generate greetings for all default voice + industry combinations
        #    These are stored in the process-level _greeting_cache in voice_agent.py
        #    so every first session gets instant audio from cache instead of cold-generating
        try:
            from app.voice_demo.controllers.voice_agent import (
                pregenerate_greeting, _greeting_cache
            )
            DEFAULT_VOICES = ["Sophia", "Maya", "Ananya", "Arjun", "David"]
            DEFAULT_INDUSTRIES = ["hospital", "real_estate"]
            pregen_session_id = "__warmup__"

            async def _prewarm_greetings():
                for voice_name in DEFAULT_VOICES:
                    for industry in DEFAULT_INDUSTRIES:
                        cache_key = (voice_name.lower(), "English", industry)
                        if cache_key not in _greeting_cache:
                            try:
                                await pregenerate_greeting(pregen_session_id, industry, "English", voice_name)
                                logger.info(f"[WARMUP-PREGEN] Pre-generated greeting: voice={voice_name} industry={industry}")
                            except Exception as pregen_err:
                                logger.warning(f"[WARMUP-PREGEN] Non-fatal: {voice_name}/{industry}: {pregen_err}")

            asyncio.create_task(_prewarm_greetings())
            logger.info("[WARMUP-PREGEN] Launched background greeting pre-generation for all voice+industry combos")
        except Exception as pregen_init_err:
            logger.warning(f"[WARMUP-PREGEN] Could not start greeting pre-generation (non-fatal): {pregen_init_err}")

    except Exception as e:
        logger.error(f"[WARMUP ERROR] Failed during AI subsystems startup warmup: {e}")

    STARTUP_METRICS["boot_time_sec"] = round(time.perf_counter() - startup_start, 2)

    yield

    # Shutdown hook: Clean up pools
    if HAS_PROD_MODULES:
        try:
            logger.info("Shutting down external service connection pools...")
            from app.db.session import get_engine
            await get_engine().dispose()
            logger.info("Database connection pool disposed.")
        except Exception as e:
            logger.error(f"Error disposing database connections: {e}")


app = FastAPI(
    title=settings.APP_NAME,
    description="Backend service powering outbound cold calls and inbound support bots.",
    version="0.1.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if RequestLoggingMiddleware is not None:
    app.add_middleware(RequestLoggingMiddleware)


if AppException is not None:
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.error(f"Application error occurred: {exc.message} (status: {exc.status_code})")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "status": "error"}
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled system error occurred: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal system error occurred.", "status": "error"}
    )


if HAS_PROD_MODULES and api_v1_router is not None:
    app.include_router(api_v1_router, prefix="/api/v1")
app.include_router(voice_agent_router, prefix="/api/v1/voice-demo")


@app.get("/voice-agent")
async def voice_agent_page():
    from fastapi.responses import FileResponse
    return FileResponse("static/voice-agent.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_route():
    from fastapi.responses import Response
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/static/index.html")
