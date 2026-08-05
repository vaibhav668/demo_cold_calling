# Public AI Voice Agent Demo

This is a premium, browser-based public demonstration of our AI Voice Agent platform. Visitors can select a language, select an industry scenario, choose a voice profile, and carry out a natural, real-time voice conversation directly in their browser.

## Tech Stack
- **Backend**: FastAPI, WebSockets
- **Speech Subsystem**: Faster-Whisper (STT), Silero VAD (VAD), EdgeTTS (TTS)
- **AI Core**: Groq / OpenRouter, SentenceTransformers (MiniLM) for in-memory RAG
- **Frontend**: Dark Glassmorphism, animated AI orb, dynamic speech animations.

## Local Development
1. Set up virtual environment and dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Set API keys:
   ```bash
   export GROQ_API_KEY="your-api-key"
   ```
3. Run the development server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
