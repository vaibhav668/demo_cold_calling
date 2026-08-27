import sys
import os
import asyncio
import kokoro_onnx.config

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

# Ensure 'h' and 'hi' are in SUPPORTED_LANGUAGES
if not hasattr(kokoro_onnx.config, "SUPPORTED_LANGUAGES"):
    kokoro_onnx.config.SUPPORTED_LANGUAGES = ["a", "b", "en-us", "h", "i", "j", "z"]
elif "h" not in kokoro_onnx.config.SUPPORTED_LANGUAGES:
    kokoro_onnx.config.SUPPORTED_LANGUAGES.append("h")

from app.services.speech.tts.kokoro_provider import KokoroProvider

async def test_native_hindi_synthesis():
    kp = KokoroProvider()
    model = await kp._get_model(kp.model_dir, kp.onnx_path, kp.voices_path)
    tok = model.tokenizer

    devanagari_text = "नमस्ते! मैं सिटीकेयर हॉस्पिटल से Maya बात कर रही हूँ। क्या मैं आपका नाम जान सकती हूँ?"
    print("=" * 75)
    print("Testing Native Devanagari Hindi Synthesis with espeak-ng lang='hi' mapped to lang='h':")
    print("=" * 75)

    _orig_phonemize = getattr(tok, "_orig_phonemize", tok.phonemize)
    
    # Configure model tokenizer to use espeak-ng 'hi' for lang='h' on raw Devanagari!
    def _native_hi_phonemize(txt, lg):
        if lg == "h":
            # Use native espeak-ng Hindi phonemizer on Devanagari script!
            return _orig_phonemize(txt, "hi")
        return _orig_phonemize(txt, lg)
    tok.phonemize = _native_hi_phonemize

    voice_style = model.get_voice_style("hf_beta")
    samples, sr = model.create(devanagari_text, voice=voice_style, speed=1.1, lang="h")
    dur = len(samples) / sr

    print(f"✓ SUCCESSFULLY SYNTHESIZED {len(samples)} SAMPLES ({dur:.2f}s AUDIO)!")
    print(f"✓ Native Indian IPA Phonemes passed to Kokoro:\n  '{_orig_phonemize(devanagari_text, 'hi')}'")

if __name__ == "__main__":
    asyncio.run(test_native_hindi_synthesis())
