import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.config import settings
from app.voice_demo.controllers.voice_agent import pregenerate_greeting, _greeting_cache, get_greeting_text

async def verify():
    print("=" * 60)
    print("       VERIFYING SVARA PRE-GENERATION FIX")
    print("=" * 60)
    
    print(f"Active TTS Provider: {settings.TTS_PROVIDER}")
    
    # Test pregeneration for Sophia, Maya, Ananya in English, Hindi, Telugu
    test_cases = [
        ("hospital", "English", "Sophia", "Female"),
        ("hospital", "English", "Maya", "Female"),
        ("hospital", "Hindi", "Maya", "Female"),
        ("hospital", "Telugu", "Maya", "Female"),
        ("real_estate", "English", "Ananya", "Female"),
        ("hospital", "English", "Arjun", "Male"),
        ("real_estate", "Hindi", "David", "Male"),
    ]

    for ind, lang, persona, gender in test_cases:
        session_id = f"test_{persona}_{lang}_{ind}"
        print(f"\n---> Testing pregenerate_greeting: persona={persona}, lang={lang}, industry={ind}")
        success = await pregenerate_greeting(session_id, ind, lang, persona, gender=gender)
        
        provider_name = settings.TTS_PROVIDER.lower().strip()
        cache_key = (provider_name, persona.lower(), lang.lower(), ind.lower(), gender.lower())
        
        if success and cache_key in _greeting_cache:
            frames = len(_greeting_cache[cache_key])
            total_bytes = sum(len(c) for c in _greeting_cache[cache_key])
            print(f"  ✓ SUCCESS: Generated & Cached key={cache_key} ({frames} frames, {total_bytes} bytes)")
        else:
            print(f"  ✗ FAILED: success={success}, key_in_cache={cache_key in _greeting_cache}")

    print("\n" + "=" * 60)
    print(f"TOTAL CACHED GREETINGS: {len(_greeting_cache)} items")
    for key, chunks in _greeting_cache.items():
        print(f"  - Key: {key} -> {len(chunks)} audio frames ({sum(len(c) for c in chunks)} bytes)")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(verify())
