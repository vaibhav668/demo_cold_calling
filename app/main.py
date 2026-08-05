import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.core.telemetry import STARTUP_METRICS
from app.services.rag_service import RAGService
from app.voice_demo.controllers.voice_agent import router as voice_agent_router

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

    # Initialize ChromaDB and seed RAG collections
    logger.info("Initializing in-memory ChromaDB RAG collection...")
    try:
        rag = RAGService()
        await rag.initialize_collection()
    except Exception as e:
        logger.error(f"Failed to auto-initialize RAG collection: {e}")

    # Eagerly warm up AI pipeline services
    try:
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

        # 3. Warm up TTS Provider
        t_t0 = time.perf_counter()
        from app.services.tts_service import VoiceService
        tts = VoiceService()
        async for _ in tts.stream_speech("Hello"):
            break
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

    except Exception as e:
        logger.error(f"[WARMUP ERROR] Failed during AI subsystems startup warmup: {e}")

    STARTUP_METRICS["boot_time_sec"] = round(time.perf_counter() - startup_start, 2)

    yield
    logger.info("Application shutdown complete.")

app = FastAPI(
    title=settings.APP_NAME,
    description="Public browser demo server for AI Cold Calling Agent.",
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

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled system error occurred: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal system error occurred.", "status": "error"}
    )

# Include Voice Demo APIs
app.include_router(voice_agent_router, prefix="/api/v1/voice-demo")

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon_route():
    from fastapi.responses import Response
    return Response(status_code=204)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/static/index.html")
