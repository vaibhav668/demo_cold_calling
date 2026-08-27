import sys
import os
import time
import asyncio
import wave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.config import settings
from app.services.speech.tts.svara_provider import SvaraProvider

TEST_SCENARIOS = [
    # (Scenario, Language, AgentPersona, Text)
    ("1_greeting", "English", "Sophia", "Hi, this is Sophia from CityCare Hospital! <warm>"),
    ("1_greeting", "Hindi", "Maya", "नमस्ते! मैं सिटीकेयर हॉस्पिटल से माया बात कर रही हूँ। <warm>"),
    ("1_greeting", "Hinglish", "Maya", "Namaste, main CityCare Hospital se Maya baat kar rahi hoon. <warm>"),
    ("1_greeting", "Telugu", "Maya", "నమస్కారం! నేను స్కైలైన్ డెవలపర్స్ నుండి మాయ మాట్లాడుతున్నాను. <warm>"),
    ("1_greeting", "Telugu-English", "Maya", "Namaskaram! Nenu Skyline Developers nundi Maya matladutunnanu. <warm>"),

    ("2_name", "English", "Sophia", "May I know whom I'm speaking with? <clear>"),
    ("2_name", "Hindi", "Maya", "क्या मैं आपका शुभ नाम जान सकती हूँ? <clear>"),
    ("2_name", "Hinglish", "Maya", "Kya main aapka naam jaan sakti hoon? <clear>"),
    ("2_name", "Telugu", "Maya", "మీ పేరు తెలుసుకోవచ్చా? <clear>"),
    ("2_name", "Telugu-English", "Maya", "Mee peru telusukovachha? <clear>"),

    ("3_question", "English", "Sophia", "Would you prefer a morning or afternoon appointment tomorrow? <clear>"),
    ("3_question", "Hindi", "Maya", "क्या आप कल सुबह या दोपहर का अपॉइंटमेंट पसंद करेंगे? <clear>"),
    ("3_question", "Hinglish", "Maya", "Kya aap kal morning ya afternoon appointment prefer karenge? <clear>"),
    ("3_question", "Telugu", "Maya", "మీరు రేపు ఉదయం లేదా మధ్యాహ్నం అపాయింట్‌మెంట్ కావాలనుకుంటున్నారా? <clear>"),
    ("3_question", "Telugu-English", "Maya", "Meeru repu morning leda afternoon appointment prefer chestara? <clear>"),

    ("4_confirmation", "English", "Sophia", "Great, Vaibhav! Your appointment has been successfully confirmed for tomorrow at 11 AM with Dr. Sharma. <happy>"),
    ("4_confirmation", "Hindi", "Maya", "बहुत बढ़िया, वैभव! आपकी अपॉइंटमेंट कल 11 बजे डॉक्टर शर्मा के साथ कन्फर्म हो गई है। <happy>"),
    ("4_confirmation", "Hinglish", "Maya", "Bahut badhiya, Vaibhav! Aapki appointment kal 11 baje Dr. Sharma ke saath confirm ho gayi hai. <happy>"),
    ("4_confirmation", "Telugu", "Maya", "చాలా మంచిది, వైభవ్! మీ అపాయింట్‌మెంట్ రేపు ఉదయం 11 గంటలకు డాక్టర్ శర్మ గారితో ఖరారైంది. <happy>"),
    ("4_confirmation", "Telugu-English", "Maya", "Chala manchidi, Vaibhav! Mee appointment repu morning 11 AM ki Dr. Sharma garitho confirm ayindi. <happy>"),

    ("5_cancellation", "English", "Sophia", "Your appointment has been successfully cancelled, Vaibhav. <clear>"),
    ("5_cancellation", "Hindi", "Maya", "वैभव, आपकी अपॉइंटमेंट सफलतापूर्वक रद्द कर दी गई है। <clear>"),
    ("5_cancellation", "Hinglish", "Maya", "Aapki appointment successfully cancel ho gayi hai, Vaibhav. <clear>"),
    ("5_cancellation", "Telugu", "Maya", "వైభవ్, మీ అపాయింట్‌మెంట్ విజయవంతంగా రద్దయింది. <clear>"),
    ("5_cancellation", "Telugu-English", "Maya", "Mee appointment successfully cancel ayindi, Vaibhav. <clear>"),

    ("6_rescheduling", "English", "Sophia", "Your appointment has been rescheduled to next Monday at 2 PM with Dr. Sharma. <clear>"),
    ("6_rescheduling", "Hindi", "Maya", "आपकी अपॉइंटमेंट अगले सोमवार दोपहर 2 बजे के लिए रीशेड्यूल कर दी गई है। <clear>"),
    ("6_rescheduling", "Hinglish", "Maya", "Aapki appointment next Monday 2 PM ko reschedule kar di gayi hai. <clear>"),
    ("6_rescheduling", "Telugu", "Maya", "మీ అపాయింట్‌మెంట్ వచ్చే సోమవారం మధ్యాహ్నం 2 గంటలకు మార్చబడింది. <clear>"),
    ("6_rescheduling", "Telugu-English", "Maya", "Mee appointment next Monday afternoon 2 PM ki reschedule chesam. <clear>"),

    ("7_empathy", "English", "Sophia", "I completely understand your concern, Vaibhav. Let me help you right away. <warm>"),
    ("7_empathy", "Hindi", "Maya", "मैं आपकी बात पूरी तरह समझती हूँ, वैभव। मैं तुरंत आपकी सहायता करती हूँ। <warm>"),
    ("7_empathy", "Hinglish", "Maya", "Main aapki baat bilkul samajhti hoon, Vaibhav. Main abhi aapki help karti hoon. <warm>"),
    ("7_empathy", "Telugu", "Maya", "నేను మీ పరిస్థితిని పూర్తిగా అర్థం చేసుకున్నాను, వైభవ్. నేను మీకు సహాయం చేస్తాను. <warm>"),
    ("7_empathy", "Telugu-English", "Maya", "Nenu mee situation ni purtiga artham chesukuntanau, Vaibhav. Nenu meeku help chestanu. <warm>"),

    ("8_sales_pitch", "English", "Ananya", "Skyline Developers offers luxury 3 BHK apartments in Gachibowli with world-class amenities! <happy>"),
    ("8_sales_pitch", "Hindi", "Ananya", "स्काईलाइन डेवलपर्स गचीबोवली में प्रीमियम 3 बीएचके फ्लैट्स और आधुनिक सुविधाएं दे रहे हैं! <happy>"),
    ("8_sales_pitch", "Hinglish", "Ananya", "Skyline Developers Gachibowli mein premium 3 BHK apartments offer kar rahe hain with luxury amenities! <happy>"),
    ("8_sales_pitch", "Telugu", "Ananya", "స్కైలైన్ డెవలపర్స్ గచ్చిబౌలిలో ప్రీమియం 3 BHK ఇళ్లను అన్ని వసతులతో అందిస్తున్నారు! <happy>"),
    ("8_sales_pitch", "Telugu-English", "Ananya", "Skyline Developers Gachibowli lo premium 3 BHK apartments offer chestunnaru with top amenities! <happy>"),

    ("9_closing", "English", "Sophia", "Thank you for your time, Vaibhav. Have a wonderful day! Goodbye. <warm>"),
    ("9_closing", "Hindi", "Maya", "समय देने के लिए धन्यवाद, वैभव। आपका दिन शुभ हो! नमस्ते। <warm>"),
    ("9_closing", "Hinglish", "Maya", "Waqt dene ke liye Dhanyavaad, Vaibhav. Have a great day! Bye. <warm>"),
    ("9_closing", "Telugu", "Maya", "మీ సమయానికి ధన్యవాదాలు, వైభవ్. మంచి రోజు అవ్వాలని కోరుకుంటున్నాను! సెలవు. <warm>"),
    ("9_closing", "Telugu-English", "Maya", "Mee time ki Dhanyavaadalu, Vaibhav. Have a great day! Bye. <warm>"),
]


async def run_diagnostics():
    out_dir = os.path.abspath("test_audio_outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("       SVARA TTS AUDIO QUALITY DIAGNOSTIC SUITE (24 kHz PCM)")
    print("=" * 80)

    svara = SvaraProvider()
    results = []

    for scenario, lang, persona, text in TEST_SCENARIOS:
        fname = f"{scenario}_{lang.lower()}_{persona.lower()}.wav"
        fpath = os.path.join(out_dir, fname)

        t_start = time.perf_counter()
        pcm_chunks = []
        async for chunk in svara.stream_speech(text, language=lang, voice_config={"persona_name": persona}):
            pcm_chunks.append(chunk)

        total_pcm = b"".join(pcm_chunks)
        synth_time = (time.perf_counter() - t_start) * 1000.0
        audio_dur_sec = len(total_pcm) / (24000 * 2) if total_pcm else 0.0
        rtf = (synth_time / 1000.0) / audio_dur_sec if audio_dur_sec > 0 else 0.0

        # Save to 24kHz 16-bit linear PCM WAV file for manual listening
        if total_pcm:
            with wave.open(fpath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(total_pcm)

        results.append({
            "scenario": scenario,
            "lang": lang,
            "persona": persona,
            "bytes": len(total_pcm),
            "dur_sec": audio_dur_sec,
            "synth_ms": synth_time,
            "rtf": rtf,
            "file": fname
        })

        print(f"[{scenario:15}] lang={lang:14} persona={persona:7} dur={audio_dur_sec:4.2f}s synth={synth_time:6.1f}ms RTF={rtf:4.2f}x -> {fname}")

    print("=" * 80)
    print(f"✓ GENERATED {len(results)} HIGH-FIDELITY 24kHz WAV SAMPLES IN: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_diagnostics())
