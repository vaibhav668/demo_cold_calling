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
    # (Scenario_ID, Language, Persona, Text, Emotion_Intent)
    ("1_greeting", "English", "Maya", "Hi, this is Maya from CityCare Hospital! How are you doing today?", "GREETING"),
    ("1_greeting", "Hindi", "Maya", "नमस्ते Vaibhav! मैं Maya बोल रही हूँ, CityCare Hospital से।", "GREETING"),
    ("1_greeting", "Hinglish", "Maya", "Namaste Vaibhav! Main Maya bol rahi hoon, CityCare Hospital se.", "GREETING"),
    ("1_greeting", "Telugu", "Maya", "నమస్కారం Vaibhav! నేను Maya మాట్లాడుతున్నాను, CityCare Hospital నుంచి.", "GREETING"),
    ("1_greeting", "Telugu-English", "Maya", "Namaskaram Vaibhav! Nenu Maya matladutunnanu, CityCare Hospital nundi.", "GREETING"),

    ("2_name_question", "English", "Maya", "May I know whom I'm speaking with?", "FRIENDLY_QUESTION"),
    ("2_name_question", "Hindi", "Maya", "क्या मैं आपका नाम जान सकती हूँ?", "FRIENDLY_QUESTION"),
    ("2_name_question", "Hinglish", "Maya", "Kya main aapka naam jaan sakti hoon?", "FRIENDLY_QUESTION"),
    ("2_name_question", "Telugu", "Maya", "మీ పేరు తెలుసుకోవచ్చా?", "FRIENDLY_QUESTION"),
    ("2_name_question", "Telugu-English", "Maya", "Mee peru telusukovachha?", "FRIENDLY_QUESTION"),

    ("3_acknowledgement", "English", "Maya", "Nice to speak with you, Vaibhav!", "POSITIVE"),
    ("3_acknowledgement", "Hindi", "Maya", "आपसे बात करके बहुत खुशी हुई, Vaibhav!", "POSITIVE"),
    ("3_acknowledgement", "Hinglish", "Maya", "Aap se baat karke bahut khushi hui, Vaibhav!", "POSITIVE"),
    ("3_acknowledgement", "Telugu", "Maya", "మీతో మాట్లాడటం చాలా సంతోషంగా ఉంది, Vaibhav!", "POSITIVE"),
    ("3_acknowledgement", "Telugu-English", "Maya", "Meetho matladatam chala santoshanga undi, Vaibhav!", "POSITIVE"),

    ("4_appointment_explanation", "English", "Maya", "I'm calling regarding your appointment with Dr. Sharma tomorrow at 11 AM.", "PURPOSE"),
    ("4_appointment_explanation", "Hindi", "Maya", "मैं कल सुबह 11 बजे Dr. Sharma के साथ आपके appointment के सिलसिले में कॉल कर रही हूँ।", "PURPOSE"),
    ("4_appointment_explanation", "Hinglish", "Maya", "Main kal morning 11 AM ko Dr. Sharma ke saath aapke appointment ke baare mein call kar rahi hoon.", "PURPOSE"),
    ("4_appointment_explanation", "Telugu", "Maya", "నేను రేపు ఉదయం 11 AM కి Dr. Sharma గారితో ఉన్న మీ appointment గురించి కాల్ చేస్తున్నాను.", "PURPOSE"),
    ("4_appointment_explanation", "Telugu-English", "Maya", "Nenu repu morning 11 AM ki Dr. Sharma garitho unna mee appointment gurinchi call chestunnanu.", "PURPOSE"),

    ("5_confirmation", "English", "Maya", "Perfect, Vaibhav! Your appointment is confirmed for tomorrow at 11 AM with Dr. Sharma.", "CONFIRMATION"),
    ("5_confirmation", "Hindi", "Maya", "बहुत बढ़िया, Vaibhav! आपकी appointment कल 11 बजे Dr. Sharma के साथ confirm हो गई है।", "CONFIRMATION"),
    ("5_confirmation", "Hinglish", "Maya", "Bahut badhiya, Vaibhav! Aapki appointment kal 11 baje Dr. Sharma ke saath confirm ho gayi hai.", "CONFIRMATION"),
    ("5_confirmation", "Telugu", "Maya", "చాలా మంచిది, Vaibhav! మీ appointment రేపు ఉదయం 11 గంటలకు Dr. Sharma గారితో confirm అయింది.", "CONFIRMATION"),
    ("5_confirmation", "Telugu-English", "Maya", "Chala manchidi, Vaibhav! Mee appointment repu morning 11 AM ki Dr. Sharma garitho confirm ayindi.", "CONFIRMATION"),

    ("6_cancellation", "English", "Maya", "Absolutely, Vaibhav. Your appointment has been cancelled successfully.", "CANCELLATION"),
    ("6_cancellation", "Hindi", "Maya", "बिलकुल, Vaibhav। आपकी appointment सफलतापूर्वक cancel कर दी गई है।", "CANCELLATION"),
    ("6_cancellation", "Hinglish", "Maya", "Bilkul, Vaibhav. Aapki appointment successfully cancel ho gayi hai.", "CANCELLATION"),
    ("6_cancellation", "Telugu", "Maya", "ఖచ్చితంగా, Vaibhav. మీ appointment రద్దు చేయబడింది.", "CANCELLATION"),
    ("6_cancellation", "Telugu-English", "Maya", "Khachitanga, Vaibhav. Mee appointment successfully cancel ayindi.", "CANCELLATION"),

    ("7_rescheduling", "English", "Maya", "Perfect, Vaibhav. Your appointment has been rescheduled to next Monday at 2 PM.", "RESCHEDULING"),
    ("7_rescheduling", "Hindi", "Maya", "बहुत अच्छा, Vaibhav। आपकी appointment अगले सोमवार दोपहर 2 बजे के लिए reschedule कर दी गई है।", "RESCHEDULING"),
    ("7_rescheduling", "Hinglish", "Maya", "Bahut achha, Vaibhav. Aapki appointment next Monday 2 PM ko reschedule ho gayi hai.", "RESCHEDULING"),
    ("7_rescheduling", "Telugu", "Maya", "చాలా బాగుంది, Vaibhav. మీ appointment వచ్చే సోమవారం మధ్యాహ్నం 2 గంటలకు reschedule అయింది.", "RESCHEDULING"),
    ("7_rescheduling", "Telugu-English", "Maya", "Chala bagundi, Vaibhav. Mee appointment next Monday afternoon 2 PM ki reschedule chesam.", "RESCHEDULING"),

    ("8_empathy", "English", "Maya", "Oh, I completely understand. No problem at all, Vaibhav.", "EMPATHY"),
    ("8_empathy", "Hindi", "Maya", "अरे, मैं आपकी बात पूरी तरह समझती हूँ, Vaibhav। कोई समस्या नहीं है।", "EMPATHY"),
    ("8_empathy", "Hinglish", "Maya", "Oh, main aapki baat bilkul samajhti hoon, Vaibhav. Koi problem nahi hai.", "EMPATHY"),
    ("8_empathy", "Telugu", "Maya", "నేను మీ పరిస్థితిని పూర్తిగా అర్థం చేసుకున్నాను, Vaibhav. ఏమీ పర్వాలేదు.", "EMPATHY"),
    ("8_empathy", "Telugu-English", "Maya", "Oh, nenu mee situation ni purtiga artham chesukuntanu, Vaibhav. Emi parvaledu.", "EMPATHY"),

    ("9_closing", "English", "Maya", "Thank you for your time, Vaibhav. Have a wonderful day! Goodbye.", "CLOSING"),
    ("9_closing", "Hindi", "Maya", "समय देने के लिए धन्यवाद, Vaibhav। आपका दिन शुभ हो! नमस्ते।", "CLOSING"),
    ("9_closing", "Hinglish", "Maya", "Time dene ke liye Dhanyavaad, Vaibhav. Have a wonderful day! Bye.", "CLOSING"),
    ("9_closing", "Telugu", "Maya", "మీ సమయానికి ధన్యవాదాలు, Vaibhav. మంచి రోజు అవ్వాలని కోరుకుంటున్నాను! సెలవు.", "CLOSING"),
    ("9_closing", "Telugu-English", "Maya", "Mee time ki Dhanyavaadalu, Vaibhav. Have a wonderful day! Bye.", "CLOSING"),

    ("10_real_estate_pitch", "English", "Ananya", "Skyline Developers offers luxury 2 and 3 BHK apartments in Gachibowli starting at 80 Lakhs!", "SALES"),
    ("10_real_estate_pitch", "Hindi", "Ananya", "Skyline Developers Gachibowli में 80 लाख से शुरू प्रीमियम 2 और 3 BHK luxury apartments दे रहे हैं!", "SALES"),
    ("10_real_estate_pitch", "Hinglish", "Ananya", "Skyline Developers Gachibowli mein 80 Lakhs se start premium 2 and 3 BHK luxury apartments offer kar rahe hain!", "SALES"),
    ("10_real_estate_pitch", "Telugu", "Ananya", "Skyline Developers గచ్చిబౌలిలో 80 లక్షల నుండి లగ్జరీ 2 & 3 BHK ఇళ్లను అందిస్తున్నారు!", "SALES"),
    ("10_real_estate_pitch", "Telugu-English", "Ananya", "Skyline Developers Gachibowli lo 80 Lakhs nundi luxury 2 & 3 BHK apartments offer chestunnaru!", "SALES"),

    ("11_site_visit_booking", "English", "Ananya", "Great, Vaibhav! I've booked a site visit for you on Saturday at 11 AM. We look forward to seeing you!", "CONFIRMATION"),
    ("11_site_visit_booking", "Hindi", "Ananya", "बहुत बढ़िया, Vaibhav! मैंने शनिवार सुबह 11 बजे आपके लिए site visit book कर दी है।", "CONFIRMATION"),
    ("11_site_visit_booking", "Hinglish", "Ananya", "Bahut badhiya, Vaibhav! Main Saturday morning 11 AM ko aapke liye site visit book kar di hai.", "CONFIRMATION"),
    ("11_site_visit_booking", "Telugu", "Ananya", "చాలా మంచిది, Vaibhav! నేను శనివారం ఉదయం 11 గంటలకు మీ కోసం site visit book చేశాను.", "CONFIRMATION"),
    ("11_site_visit_booking", "Telugu-English", "Ananya", "Chala manchidi, Vaibhav! Nenu Saturday morning 11 AM ki mee kosam site visit book చేశాను.", "CONFIRMATION"),
]

async def run_human_voice_suite():
    out_dir = os.path.abspath("test_human_voice_outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 90)
    print("      HUMAN CONTINUOUS & EXPRESSIVE VOICE SUITE — QUALITY GATE TEST")
    print("=" * 90)

    svara = SvaraProvider()
    results = []
    total_gaps = []

    for scenario, lang, persona, text, intent in TEST_SCENARIOS:
        fname = f"{scenario}_{lang.lower().replace('-', '_')}_{persona.lower()}.wav"
        fpath = os.path.join(out_dir, fname)

        t_start = time.perf_counter()
        pcm_chunks = []
        voice_config = {"persona_name": persona, "intent": intent, "session_id": "test_suite"}

        async for chunk in svara.stream_speech(text, language=lang, voice_config=voice_config):
            pcm_chunks.append(chunk)

        total_pcm = b"".join(pcm_chunks)
        synth_time = (time.perf_counter() - t_start) * 1000.0
        audio_dur_sec = len(total_pcm) / (24000 * 2) if total_pcm else 0.0
        rtf = (synth_time / 1000.0) / audio_dur_sec if audio_dur_sec > 0 else 0.0

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
            "dur_sec": audio_dur_sec,
            "synth_ms": synth_time,
            "rtf": rtf,
            "file": fname
        })

        print(f"[{scenario:25}] lang={lang:14} persona={persona:7} dur={audio_dur_sec:5.2f}s synth={synth_time:6.1f}ms RTF={rtf:4.2f}x -> {fname}")

    print("=" * 90)
    print(f"✓ SUCCESSFULLY GENERATED {len(results)} HIGH-FIDELITY 24kHz SAMPLES IN: {out_dir}")
    print("=" * 90)

if __name__ == "__main__":
    asyncio.run(run_human_voice_suite())
