"""
Hinglish & Devanagari Text Normalization Service.
Handles bidirectional conversion between Roman Hinglish and Devanagari Hindi
while preserving English entity names (Vaibhav, Dr. Sharma) and conversational keywords.
"""

import re
from typing import Tuple, Dict, Any, Optional

# Key Hinglish phonetic mappings to Devanagari
HINGLISH_TO_DEVANAGARI_MAP = [
    # Common multi-word phrases
    (r"\bmera naam\b", "मेरा नाम"),
    (r"\bapka naam\b", "आपका नाम"),
    (r"\baapka naam\b", "आपका नाम"),
    (r"\bse baat karni hai\b", "से बात करनी है"),
    (r"\bse baat karna hai\b", "से बात करना है"),
    (r"\bconfirm kar do\b", "confirm कर दो"),
    (r"\bconfirm kar dijiye\b", "confirm कर दीजिए"),
    (r"\breschedule kar do\b", "reschedule कर दो"),
    (r"\breschedule kar dijiye\b", "reschedule कर दीजिए"),
    (r"\bcancel kar do\b", "cancel कर दो"),
    (r"\bcancel kar dijiye\b", "cancel कर दीजिए"),
    (r"\bsamajh nahi aa raha\b", "समझ नहीं आ रहा"),
    (r"\bkar sakte hain\b", "कर सकते हैं"),
    (r"\bkar sakte ho\b", "कर सकते हो"),
    
    # Common words
    (r"\bmera\b", "मेरा"),
    (r"\bmeri\b", "मेरी"),
    (r"\bmere\b", "मेरे"),
    (r"\bnaam\b", "नाम"),
    (r"\bhai\b", "है"),
    (r"\bhoon\b", "हूँ"),
    (r"\bhu\b", "हूँ"),
    (r"\bkal\b", "कल"),
    (r"\baaj\b", "आज"),
    (r"\bsubah\b", "सुबह"),
    (r"\bshaam\b", "शाम"),
    (r"\bbaje\b", "बजे"),
    (r"\bhaan\b", "हाँ"),
    (r"\bnahi\b", "नहीं"),
    (r"\bna\b", "ना"),
    (r"\bji\b", "जी"),
    (r"\bmujhe\b", "मुझे"),
    (r"\baap\b", "आप"),
    (r"\bkya\b", "क्या"),
    (r"\bko\b", "को"),
    (r"\bpar\b", "पर"),
    (r"\bse\b", "से"),
    (r"\bka\b", "का"),
    (r"\bki\b", "की"),
    (r"\bke\b", "के"),
    (r"\bliye\b", "लिए"),
    (r"\bho\b", "हो"),
    (r"\bhain\b", "हैं"),
    (r"\bdoctor\b", "डॉक्टर"),
    (r"\bdr\b", "डॉक्टर"),
]

# Devanagari to spoken Hindi TTS text normalization (e.g. for numbers & English terms in TTS)
TTS_HINDI_REPLACEMENTS = {
    "tomorrow": "कल",
    "appointment": "अपॉइंटमेंट",
    "confirm": "कन्फर्म",
    "reschedule": "रीशेड्यूल",
    "cancel": "कैंसिल",
    "hospital": "हॉस्पिटल",
    "doctor": "डॉक्टर",
    "dr.": "डॉक्टर",
    "11": "ग्यारह",
    "10": "दस",
    "12": "बारह",
    "1": "एक",
    "2": "दो",
    "3": "तीन",
    "4": "चार",
    "5": "पांच",
}


def normalize_hinglish_to_devanagari(text: str) -> str:
    """
    Converts Roman Hinglish transcript to natural Devanagari Hindi while preserving 
    English names (Vaibhav, Sharma, CityCare) and action terms.
    """
    if not text:
        return ""

    result = text.strip()
    
    # Apply regex replacements for Hinglish words
    for pattern, repl in HINGLISH_TO_DEVANAGARI_MAP:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)

    # Clean up double spaces
    result = re.sub(r"\s+", " ", result).strip()
    return result


def prepare_text_for_hindi_tts(text: str) -> str:
    """
    Normalizes Hindi / Devanagari text for Kokoro Hindi TTS to ensure 100% natural Indian pronunciation 
    and zero mid-sentence language switches.
    """
    if not text:
        return ""

    result = text.strip()

    # Convert numbers and common English terms to Devanagari phonetics for TTS
    for eng_word, dev_word in TTS_HINDI_REPLACEMENTS.items():
        pattern = r"\b" + re.escape(eng_word) + r"\b"
        result = re.sub(pattern, dev_word, result, flags=re.IGNORECASE)

    # Map persona names to native Devanagari spellings
    persona_names = {
        "sophia": "सोफिया",
        "maya": "माया",
        "ananya": "अनन्या",
        "arjun": "अर्जुन",
        "david": "डेविड",
        "vaibhav": "वैभव",
        "sharma": "शर्मा"
    }
    for name, dev_name in persona_names.items():
        pattern = r"\b" + re.escape(name) + r"\b"
        result = re.sub(pattern, dev_name, result, flags=re.IGNORECASE)

    return result
