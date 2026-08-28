import uuid
import json
import re
import asyncio
from typing import Tuple, List, Dict, Any, Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.session_manager import SessionManager
from app.services.llm_service import LLMService
from app.services.prompt_service import PromptService
from app.services.rag_service import RAGService
HAS_DB = True
try:
    from app.repositories.call_log import CallLogRepository
    from app.models.call_log import CallLog
except ImportError:
    HAS_DB = False
    CallLogRepository = None
    CallLog = None

from app.core.logging import logger

class ConversationEngine:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.session_manager = SessionManager()
        self.llm_service = LLMService()
        self.prompt_service = PromptService(db)
        self.rag_service = RAGService()
        self.call_log_repo = CallLogRepository(db) if CallLogRepository is not None else None

    def _get_tools_schema(self) -> List[Dict[str, Any]]:
        """Define schemas for conversational tools available to LLaMA models."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "book_appointment",
                    "description": "Schedule a customer appointment or reservation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "ISO date string (YYYY-MM-DD)"},
                            "time": {"type": "string", "description": "Time string (HH:MM)"}
                        },
                        "required": ["date", "time"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "transfer_to_human",
                    "description": "Transfer the call to a human operator or support representative.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_knowledge",
                    "description": "Query the campaign knowledge database for specific business details or answers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Specific query term"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "confirm_appointment",
                    "description": "Confirm the customer is attending the scheduled appointment.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "reschedule_appointment",
                    "description": "Trigger the rescheduling workflow when a customer explicitly requests a change.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "new_date": {"type": "string", "description": "Proposed new date (YYYY-MM-DD)"},
                            "new_time": {"type": "string", "description": "Proposed new time (HH:MM)"}
                        },
                        "required": ["new_date", "new_time"]
                    }
                }
            }
        ]

    async def process_turn_stream(
        self,
        call_id: str,
        campaign_id: uuid.UUID,
        customer_id: uuid.UUID,
        user_text: str
    ) -> AsyncGenerator[Tuple[Optional[str], bool, bool], None]:
        """
        Streaming turn execution loop.
        Yields (text_token, should_hangup, should_transfer) progressively.
        """
        history = await self.session_manager.get_message_history(call_id)
        state = await self.session_manager.get_session_state(call_id) or "greeting"

        # 1. Initialize session if empty
        if not history:
            compiled_prompt, _ = await self.prompt_service.build_prompt(
                campaign_id=campaign_id,
                customer_id=customer_id,
                rag_query=user_text,
                session_id=call_id
            )
            metadata = await self.session_manager.get_session_metadata(call_id)
            if metadata and "language" in metadata:
                lang = metadata["language"]
                if lang == "Hindi":
                    compiled_prompt += (
                        "\n\n### LANGUAGE GUIDELINE\n"
                        "IMPORTANT: Speak only in Hindi. Translate all concepts, questions, and responses to Hindi naturally. "
                        "Do NOT use English or Roman script. Use Devanagari script for output."
                    )
                elif lang == "Telugu":
                    compiled_prompt += (
                        "\n\n### LANGUAGE GUIDELINE\n"
                        "IMPORTANT: Speak only in Telugu. Translate all concepts, questions, and responses to Telugu naturally. "
                        "Do NOT use English or Roman script. Use Telugu script for output."
                    )
            history.append({"role": "system", "content": compiled_prompt})
            await self.session_manager.append_message(call_id, history[-1])

        # 2. Append user input
        is_greeting = (user_text == "[CALL_START]")
        if is_greeting:
            history.append({
                "role": "system",
                "content": (
                    "[CALL_START] The call just connected. Greet the customer warmly "
                    "by name if available, introduce yourself as the AI assistant, "
                    "and state the purpose of the call concisely. "
                    "Do NOT call any tools yet. Speak naturally as if starting a phone call."
                )
            })
            user_text_for_llm = "[Please begin with your greeting now.]"
        else:
            user_text_for_llm = user_text

        user_turn = {"role": "user", "content": user_text_for_llm}
        history.append(user_turn)
        await self.session_manager.append_message(call_id, user_turn)

        # 3. Agentic tool-call loop
        should_hangup = False
        should_transfer = False
        loop_limit = 3
        full_content_accumulator = []
        active_tools = None if is_greeting else self._get_tools_schema()

        while loop_limit > 0:
            tool_calls_detected = None

            async for text_chunk, t_calls in self.llm_service.generate_completion_stream(history, active_tools):
                if t_calls:
                    tool_calls_detected = t_calls
                    break
                if text_chunk:
                    full_content_accumulator.append(text_chunk)
                    yield text_chunk, False, False

            if not tool_calls_detected:
                # Normal text response complete
                break

            # LLM requested tool execution
            tool_calls_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_detected
            }
            history.append(tool_calls_message)
            await self.session_manager.append_message(call_id, tool_calls_message)

            for tool_call in tool_calls_detected:
                tool_id = tool_call.get("id")
                func_data = tool_call.get("function", {})
                func_name = func_data.get("name")
                args = {}
                try:
                    args = json.loads(func_data.get("arguments", "{}"))
                except Exception:
                    pass

                tool_result_content = ""
                if func_name == "book_appointment":
                    state = "appointment_booked"
                    await self.session_manager.update_session_state(call_id, state)
                    tool_result_content = f"Appointment successfully scheduled for {args.get('date')} at {args.get('time')}."
                elif func_name == "transfer_to_human":
                    state = "escalated"
                    await self.session_manager.update_session_state(call_id, state)
                    should_transfer = True
                    tool_result_content = "Call transfer successfully initiated."
                elif func_name == "lookup_knowledge":
                    query = args.get("query", "")
                    facts = await self.rag_service.search_knowledge(campaign_id, query, limit=2)
                    facts_list = [f["text"] for f in facts]
                    tool_result_content = json.dumps({"facts": facts_list})
                elif func_name == "confirm_appointment":
                    state = "appointment_confirmed"
                    await self.session_manager.update_session_state(call_id, state)
                    tool_result_content = "Appointment successfully confirmed in the database."
                elif func_name == "reschedule_appointment":
                    state = "appointment_rescheduled"
                    await self.session_manager.update_session_state(call_id, state)
                    tool_result_content = f"Appointment rescheduled successfully for {args.get('new_date')} at {args.get('new_time')}."
                else:
                    tool_result_content = f"Error: Tool '{func_name}' not implemented."

                tool_response = {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": func_name,
                    "content": tool_result_content
                }
                history.append(tool_response)
                await self.session_manager.append_message(call_id, tool_response)

            loop_limit -= 1

        full_text = "".join(full_content_accumulator).strip()
        if full_text:
            bot_turn = {"role": "assistant", "content": full_text}
            history.append(bot_turn)
            await self.session_manager.append_message(call_id, bot_turn)

        # Evaluate hangup condition
        low_content = full_text.lower()
        assistant_turns = sum(1 for m in history if m.get("role") == "assistant")
        FAREWELL_RE = re.compile(
            r'\b(goodbye for now|have a great day|take care, goodbye|'
            r'thanks for calling, goodbye|thank you for calling, goodbye|'
            r'have a wonderful day|is there anything else before we go)\b'
        )
        if FAREWELL_RE.search(low_content) and assistant_turns >= 2:
            should_hangup = True
        elif state == "completed":
            should_hangup = True

        if state == "escalated":
            should_transfer = True

        yield None, should_hangup, should_transfer

    async def process_turn(
        self,
        call_id: str,
        campaign_id: uuid.UUID,
        customer_id: uuid.UUID,
        user_text: str
    ) -> Tuple[str, bool, bool]:
        """Legacy turn execution helper that collects stream output into a single string."""
        accumulated_text = []
        final_hangup = False
        final_transfer = False
        async for chunk, h, t in self.process_turn_stream(call_id, campaign_id, customer_id, user_text):
            if chunk:
                accumulated_text.append(chunk)
            if h:
                final_hangup = True
            if t:
                final_transfer = True
        return "".join(accumulated_text), final_hangup, final_transfer

    async def end_call(
        self,
        call_id: str,
        campaign_id: uuid.UUID,
        customer_id: uuid.UUID,
        phone_number: str,
        duration_seconds: int
    ) -> CallLog:
        """Purge active Redis memory keys and flush completed conversation transcripts to PostgreSQL."""
        history = await self.session_manager.get_message_history(call_id)
        state = await self.session_manager.get_session_state(call_id) or "completed"
        
        exchanges = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role in ["user", "assistant"] and content:
                exchanges.append({
                    "sender": "customer" if role == "user" else "agent",
                    "text": content
                })
                
        status_val = "completed"
        if state == "escalated":
            status_val = "completed"
        elif not exchanges:
            status_val = "failed"
            
        existing_log = await self.call_log_repo.get_by_plivo_uuid(call_id)
        if existing_log:
            updated_log = await self.call_log_repo.update(existing_log, {
                "status": status_val,
                "duration_seconds": duration_seconds,
                "transcript": exchanges
            })
            await self.db.commit()
            await self.session_manager.clear_session(call_id)
            return updated_log
            
        call_log = CallLog(
            campaign_id=campaign_id,
            customer_id=customer_id,
            plivo_call_uuid=call_id,
            phone_number=phone_number,
            status=status_val,
            duration_seconds=duration_seconds,
            transcript=exchanges
        )
        
        created_log = await self.call_log_repo.create(call_log)
        await self.db.commit()
        await self.session_manager.clear_session(call_id)
        return created_log


# ─────────────────────────────────────────────────────────────────────────────
# VOICE DEMO WEBSOCKET PIPELINE SUPPORT STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES = {
    "hospital": {
        "HOSPITAL_GREETING": {
            "en": "Hi, this is {agent_name} from CityCare Hospital. May I know whom I'm speaking with?",
            "hi": "नमस्ते, मैं सिटीकेयर हॉस्पिटल से {agent_name} बात कर रही हूँ। क्या मैं आपका नाम जान सकती हूँ?",
            "te": "నమస్కారం, నేను సిటీకేర్ హాస్పిటల్ నుండి {agent_name} మాట్లాడుతున్నాను. మీ పేరు తెలుసుకోవచ్చా?"
        },
        "HOSPITAL_PURPOSE": {
            "en": "Nice to speak with you, {customer_name}. I'm calling regarding your upcoming appointment with Dr. Sharma. Would you like to confirm, cancel, or reschedule it?",
            "hi": "आपसे बात करके अच्छा लगा {customer_name}। मैं डॉक्टर शर्मा के साथ कल के आपके अपॉइंटमेंट के संबंध में कॉल कर रही हूँ। क्या आप इसे कन्फर्म करना चाहते हैं, कैंसिल करना चाहते हैं या रीशेड्यूल करना चाहते हैं?",
            "te": "మీతో మాట్లాడటం సంతోషంగా ఉంది {customer_name}. రేపు డాక్టర్ శర్మగారితో ఉన్న మీ అపాయింట్‌మెంట్ గురించి కాల్ చేస్తున్నాను. మీరు దానిని కన్ఫర్మ్ చేయాలనుకుంటున్నారా, రద్దు చేయాలనుకుంటున్నారా లేదా రీషెడ్యూల్ చేయాలనుకుంటున్నారా?"
        },
        "CONFIRM": {
            "en": "Thank you, {customer_name}! Your appointment with Dr. Sharma has been confirmed for tomorrow at 11 AM. Is there anything else I can help you with?",
            "hi": "धन्यवाद {customer_name}! आपका डॉक्टर शर्मा के साथ अपॉइंटमेंट कल सुबह 11 बजे के लिए कन्फर्म कर दिया गया है। क्या मैं आपकी किसी और चीज़ में मदद कर सकती हूँ?",
            "te": "ధన్యవాదాలు {customer_name}! డాక్టర్ శర్మగారితో రేపు ఉదయం 11 గంటలకు మీ అపాయింట్‌మెంట్ కన్ఫర్మ్ చేయబడింది. నేను మీకు ఇంకేమైనా సహాయం చేయగలనా?"
        },
        "CANCEL": {
            "en": "No problem, {customer_name}. Your appointment with Dr. Sharma has been cancelled. Is there anything else I can help you with?",
            "hi": "कोई बात नहीं {customer_name}। आपका डॉक्टर शर्मा के साथ अपॉइंटमेंट कैंसिल कर दिया गया है। क्या मैं आपकी किसी और चीज़ में मदद कर सकती हूँ?",
            "te": "పర్వాలేదండి {customer_name}. మీ అపాయింట్‌మెంట్ రద్దు చేయబడింది. నేను మీకు ఇంకేమైనా సహాయం చేయగలనా?"
        },
        "RESCHEDULE": {
            "en": "Sure, {customer_name}. What day or time would you prefer for your rescheduled appointment?",
            "hi": "ज़रूर {customer_name}। आप अपने रीशेड्यूल किए गए अपॉइंटमेंट के लिए कौन सा दिन या समय पसंद करेंगे?",
            "te": "సరే {customer_name}. మీ రీషెడ్యూల్డ్ అపాయింట్‌మెంట్ కోసం ఏ రోజు లేదా సమయం అనుకూలంగా ఉంటుంది?"
        },
        "RESCHEDULE_CONFIRM": {
            "en": "Great! Your appointment with Dr. Sharma has been rescheduled for {slot}. Is there anything else I can help you with?",
            "hi": "बढ़िया! आपका अपॉइंटमेंट {slot} के लिए रीशेड्यूल कर दिया गया है। क्या मैं आपकी किसी और चीज़ में मदद कर सकती हूँ?",
            "te": "చాలా సంతోషం! మీ అపాయింట్‌మెంట్ {slot} కి రీషెడ్యూల్ చేయబడింది. నేను మీకు ఇంకేమైనా సహాయం చేయగలనా?"
        },
        "AMBIGUOUS_YES": {
            "en": "Would you like to confirm, cancel, or reschedule the appointment?",
            "hi": "क्या आप अपॉइंटमेंट कन्फर्म करना चाहते हैं, कैंसिल करना चाहते हैं या रीशेड्यूल करना चाहते हैं?",
            "te": "మీరు అపాయింట్‌మెంట్ కన్ఫర్మ్ చేయాలనుకుంటున్నారా, రద్దు చేయాలనుకుంటున్నారా లేదా రీషెడ్యూల్ చేయాలనుకుంటున్నారా?"
        },
        "ANYTHING_ELSE": {
            "en": "Is there anything else I can help you with?",
            "hi": "क्या मैं आपकी किसी और चीज़ में मदद कर सकती हूँ?",
            "te": "నేను మీకు ఇంకేమైనా సహాయం చేయగలనా?"
        },
        "GOODBYE": {
            "en": "Thank you for your time, {customer_name}. Have a great day. Goodbye!",
            "hi": "समय देने के लिए धन्यवाद, {customer_name}। आपका दिन शुभ हो। नमस्ते!",
            "te": "మీ సమయానికి ధన్యవాదాలు, {customer_name}. మంచి రోజు అవ్వాలని కోరుకుంటున్నాను. సెలవు!"
        },
        "MEDICAL_SAFETY": {
            "en": "I can't diagnose a medical condition, but symptoms like that can require prompt medical attention. Please contact CityCare Hospital's emergency department at +91 22 5550 9999 or consult a qualified medical professional.",
            "hi": "मैं चिकित्सीय स्थिति का निदान नहीं कर सकती, लेकिन ऐसे लक्षणों के लिए तुरंत चिकित्सा ध्यान देने की आवश्यकता हो सकती है। कृपया सिटीकेयर अस्पताल के आपातकालीन विभाग +91 22 5550 9999 पर संपर्क करें या किसी योग्य डॉक्टर से परामर्श लें।",
            "te": "నేను వైద్య పరిస్థితిని నిర్ధారించలేను, కానీ అలాంటి లక్షణాలకు తక్షణ వైద్య సంరక్షణ అవసరం కావచ్చు. దయచేసి సిటీకేర్ హాస్పిటల్ ఎమర్జెన్సీ విభాగం +91 22 5550 9999ను సంప్రదించండి లేదా అర్హత కలిగిన డాక్టర్‌ను సంప్రదించండి."
        },
        "UNKNOWN": {
            "en": "I'm sorry, I don't have that information available right now.",
            "hi": "क्षमा करें, मेरे पास अभी वह जानकारी उपलब्ध नहीं है।",
            "te": "క్షమించండి, నా వద్ద ప్రస్తుతం ఆ సమాచారం అందుబాటులో లేదు."
        },
        "UNCLEAR": {
            "en": "Just regarding your appointment, would you like to confirm, reschedule, or cancel?",
            "hi": "बस आपके अपॉइंटमेंट के बारे में, क्या आप इसकी पुष्टि करना चाहते हैं, इसे रीशेड्यूल करना चाहते हैं या कैंसिल करना चाहते हैं?",
            "te": "మీ అపాయింట్‌మెంట్ గురించి, మీరు దానిని కన్ఫర్మ్ చేయాలనుకుంటున్నారా, రీషెడ్యూల్ చేయాలనుకుంటున్నారా లేదా రద్దు చేయాలనుకుంటున్నారా?"
        },
        "CLOSING": {
            "en": "Thank you for your time, {customer_name}. Have a great day. Goodbye!",
            "hi": "समय देने के लिए धन्यवाद, {customer_name}। आपका दिन शुभ हो। नमस्ते!",
            "te": "మీ సమయానికి ధన్యవాదాలు, {customer_name}. మంచి రోజు అవ్వాలని కోరుకుంటున్నాను. సెలవు!"
        },
        "REDIRECT_SMALL_TALK": {
            "en": "Just regarding your doctor's appointment scheduled for tomorrow, would you like to confirm, reschedule, or cancel it?",
            "hi": "बस कल के आपके डॉक्टर के अपॉइंटमेंट के संबंध में, क्या आप इसकी पुष्टि करना चाहते हैं, इसे रीशेड्यूल करना चाहते हैं या कैंसिल करना चाहते हैं?",
            "te": "రేపటి మీ డాక్టర్ అపాయింట్‌మెంట్‌కు సంబంధించి, మీరు దానిని కన్ఫర్మ్ చేయాలనుకుంటున్నారా, రీషెడ్యూల్ చేయాలనుకుంటున్నారా లేదా రద్దు చేయాలనుకుంటున్నారా?"
        },
        "RECOVERY": {
            "en": "Sorry, I didn't catch your name. Could you please repeat your name?",
            "hi": "क्षमा करें, मुझे आपका नाम समझ नहीं आया। क्या आप अपना नाम दोहरा सकते हैं?",
            "te": "క్షమించండి, మీ పేరు నాకు స్పష్టంగా వినపడలేదు. మరోసారి మీ పేరు చెప్పగలరా?"
        }
    },
    "real_estate": {
        "RE_GREETING": {
            "en": "Hello... May I know whom I am speaking with?",
            "hi": "नमस्ते! क्या मैं जान सकती हूँ कि मैं किससे बात कर रही हूँ?",
            "te": "నమస్కారం! నేను ఎవరితో మాట్లాడుతున్నానో తెలుసుకోవచ్చా?"
        },
        "RE_PURPOSE_INTRO": {
            "en": "Hi {customer_name}! I'm calling from Skyline Developers... We have a 2 BHK at 80 Lakhs, a 3 BHK at 1.2 Crores, and a penthouse at 2.5 Crores... Which interests you?",
            "hi": "नमस्ते {customer_name}, बात करने के लिए धन्यवाद। मैं Skyline Developers से कॉल कर रही हूँ। हमारे पास वर्तमान में तीन प्रीमियम विकल्प उपलब्ध हैं: पहला, 80 लाख में Skyline Heights पर एक लग्जरी 2 BHK; दूसरा, 1.2 करोड़ में Skyline Residency पर एक प्रीमियम 3 BHK; और तीसरा, 2.5 करोड़ में Skyline Towers पर एक डुप्लेक्स पेंटहाउस। आप इनमें से कौन सा विकल्प पसंद करेंगे?",
            "te": "నమస్కారం {customer_name}, మాట్లాడినందుకు ధన్యవాదాలు. నేను Skyline Developers నుండి కాల్ చేస్తున్నాను. మా వద్ద ప్రస్తుతం మూడు ప్రీమియం ఆప్షన్‌లు ఉన్నాయి: మొదటిది, 80 లక్షలకు Skyline Heights వద్ద లగ్జరీ 2 BHK; రెండవది, 1.2 కోట్లకు Skyline Residency వద్ద ప్రీమియం 3 BHK; మరియు మూడవది, 2.5 కోట్లకు Skyline Towers వద్ద డ్యూప్లెక్స్ పెంట్‌హౌస్. మీరు వీటిలో ఏది ఎంచుకుంటారు?"
        },
        "RE_PROPERTY_PITCH": {
            "en": "Great! Skyline Residency features premium luxury spaces with state-of-the-art amenities. Are you currently looking to buy or invest in a property?",
            "hi": "बढ़िया! Gachibowli में Skyline Residency project में 2 और 3 BHK flats 80 लाख से शुरू हैं। क्या आप अभी कोई flat खरीदने या invest करने की सोच रहे हैं?",
            "te": "చాలా మంచిది! గచ్చిబౌలిలో Skyline Residency ప్రాజెక్ట్‌లో 2 & 3 BHK అపార్ట్‌మెంట్‌లు 80 లక్షల నుండి అందుబాటులో ఉన్నాయి. మీరు ఇల్లు కొనడానికి ఆసక్తి చూపుతున్నారా?"
        },
        "RE_INTEREST_QUESTION": {
            "en": "Awesome! What kind of property type and budget range are you considering?",
            "hi": "बढ़िया! आप किस तरह की property और budget range की सोच रहे हैं?",
            "te": "చాలా సంతోషం! మీ బడ్జెట్ మరియు ఎలాంటి ఇల్లు కావాలనుకుంటున్నారో చెప్పగలరా?"
        },
        "RE_RECOMMENDATION": {
            "en": "Based on your requirements, I would highly recommend our {unit_choice} in Skyline Residency. Would you be interested in booking a site visit to check it out?",
            "hi": "आपकी पसंद के हिसाब से, मैं आपको Skyline Residency में हमारा {unit_choice} recommend करूँगी। क्या आप इसे देखने के लिए एक site visit book करना चाहेंगे?",
            "te": "మీ ప్రాధాన్యత ప్రకారం, నేను Skyline Residency లోని మా {unit_choice} ను రికమండ్ చేస్తాను. దానిని చూడటానికి సైట్ విజిట్ బుక్ చేయాలనుకుంటున్నారా?"
        },
        "RE_SITE_VISIT_CONFIRM": {
            "en": "Perfect! Your site visit is confirmed... We will send you the details soon. Goodbye!",
            "hi": "परफेक्ट! हमने आपकी पसंद की प्रॉपर्टी देखने के लिए एक साइट विजिट बुक कर दी है। हम आपको जल्द ही सारी डिटेल्स भेज देंगे। समय देने के लिए धन्यवाद। नमस्ते!",
            "te": "చాలా సంతోషం! మీరు కోరుకున్న ప్రాపర్టీని చూడటానికి సైట్ విజిట్ బుక్ చేయబడింది. మేము మీకు త్వరలోనే పూర్తి వివరాలు పంపిస్తాము. మీ సమయానికి ధన్యవాదాలు. సెలవు!"
        },
        "RE_SITE_VISIT_DECLINE": {
            "en": "No problem. Thanks for your time, {customer_name}. Have a great day. Goodbye!",
            "hi": "कोई बात नहीं। समय देने के लिए धन्यवाद, {customer_name}। आपका दिन शुभ हो। नमस्ते!",
            "te": "పర్వాలేదండి. మీ సమయానికి ధన్యవాదాలు, {customer_name}. మంచి రోజు అవ్వాలని కోరుకుంటున్నాను. సెలవు!"
        },
        "NOT_INTERESTED": {
            "en": "No problem at all. Thanks for your time, {customer_name}. Have a great day. Goodbye!",
            "hi": "कोई बात नहीं। समय देने के लिए धन्यवाद, {customer_name}। आपका दिन शुभ हो। नमस्ते!",
            "te": "పర్వాలేదండి. మీ సమయానికి ధన్యవాదాలు, {customer_name}. మంచి రోజు అవ్వాలని కోరుకుంటున్నాను. సెలవు!"
        },
        "BUSY": {
            "en": "No problem. Thanks for your time. Goodbye!",
            "hi": "कोई बात नहीं। समय देने के लिए धन्यवाद। नमस्ते!",
            "te": "పర్వాలేదండి. మీ సమయానికి ధన్యవాదాలు. సెలవు!"
        },
        "RECOVERY": {
            "en": "Sorry, I didn't catch your name. Could you please repeat your name?",
            "hi": "क्षमा करें, मुझे आपका नाम समझ नहीं आया। क्या आप अपना नाम दोहरा सकते हैं?",
            "te": "क्षमించండి, మీ పేరు నాకు స్పష్టంగా వినపడలేదు. మరోసారి మీ పేరు చెప్పగలరా?"
        }
    }
}


def clean_speech_text(text: str) -> str:
    """Sanitize output text to ensure pure direct speech for TTS, preserving spaces."""
    if not text:
        return ""
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r'#+\s+', '', text)
    text = text.replace("`", "")
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    return text


def normalize_name_transcript(text: str) -> str:
    """Clean and normalize transcript text for name slot extraction."""
    if not text:
        return ""
    return text.strip().rstrip(".,!?।")


def extract_bhk_choice(text: str, lang_key: str = "en") -> str:
    """Extracts and formats BHK / Penthouse choice from customer text."""
    if not text:
        return {
            "en": "premium 2 BHK apartment",
            "hi": "प्रीमियम 2 BHK फ्लैट",
            "te": "ప్రీమియం 2 BHK అపార్ట్‌మెంట్"
        }.get(lang_key, "premium 2 BHK apartment")
        
    t_low = text.lower()
    if any(p in t_low for p in ["penthouse", "duplex", "पेंटहाउस", "పెంట్‌హౌస్", "2.5", "2.5 crore", "2.5cr"]):
        return {
            "en": "luxury duplex Penthouse",
            "hi": "लक्जरी डुप्लेक्स पेंटहाउस",
            "te": "లగ్జరీ డ్యూప్లెక్స్ పెంట్‌హౌస్"
        }.get(lang_key, "luxury duplex Penthouse")
    elif any(p in t_low for p in ["3 bhk", "three bhk", "3bhk", "1.2", "1.2 crore", "1.2cr", "3"]):
        return {
            "en": "premium 3 BHK apartment",
            "hi": "प्रीमियम 3 BHK फ्लैट",
            "te": "ప్రీమియం 3 BHK అపార్ట్‌మెంట్"
        }.get(lang_key, "premium 3 BHK apartment")
    else:
        return {
            "en": "premium 2 BHK apartment",
            "hi": "प्रीमियम 2 BHK फ्लैट",
            "te": "ప్రీమియం 2 BHK అపార్ట్‌మెంట్"
        }.get(lang_key, "premium 2 BHK apartment")


def clean_reschedule_slot(text: str) -> Optional[str]:
    """
    Extracts and cleans the target date/time slot from a user utterance.
    E.g. "Please reschedule it for tomorrow at 11 AM" -> "tomorrow at 11 AM"
         "Can you reschedule for Friday 4 PM" -> "Friday 4 PM"
         "We're scheduled for tomorrow morning 11 am" -> "tomorrow morning 11 am"
         "Please reschedule it" -> None (user did not provide a slot)
    """
    if not text:
        return None
    raw = text.strip().rstrip(".,!?।")

    # Check if user mentioned any date/time keyword
    time_date_pattern = re.compile(
        r'\b('
        r'monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
        r'mon|tue|wed|thu|fri|sat|sun|'
        r'today|tomorrow|day after tomorrow|yesterday|tonight|'
        r'morning|afternoon|evening|night|noon|'
        r'next week|this week|next month|next|'
        r'\d{1,2}\s*(?::\d{2})?\s*(?:am|pm|o\'clock|oclock)?|'
        r'\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)|'
        r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}|'
        r'कल|आज|परसों|सुबह|दोपहर|शाम|रात|बजे|\d{1,2}\s*बजे|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|अगले|अगला|'
        r'రేపు|ఈరోజు|ఎల్లుండి|ఉదయం|మధ్యాహ్నం|సాయంత్రం|రాత్రి|గంటలకు|\d{1,2}\s*గంటలకు|సోమవారం|మంగళవారం|బుధవారం|గురువారం|శుక్రవారం|శనివారం|ఆదివారం'
        r')\b',
        re.IGNORECASE
    )

    has_time_date = bool(time_date_pattern.search(raw))
    if not has_time_date:
        return None

    # Strip conversational prefixes
    cleaned = raw
    prefixes = [
        r"^(?:can\s+you\s+please|could\s+you\s+please|please|kindly)\s+",
        r"^(?:i\s+want\s+to|i\s+would\s+like\s+to|i\'d\s+like\s+to|i\s+need\s+to|can\s+we|could\s+we|can\s+you|could\s+you)\s+",
        r"^(?:we\'re\s+scheduled\s+for|were\s+scheduled\s+for|we\s+are\s+scheduled\s+for)\s+",
        r"^(?:reschedule\s+it\s+for|reschedule\s+for|reschedule\s+it\s+to|reschedule\s+to|reschedule\s+it|reschedule)\s+",
        r"^(?:schedule\s+it\s+for|schedule\s+for|schedule\s+it\s+to|schedule\s+to|schedule\s+it|schedule)\s+",
        r"^(?:change\s+it\s+to|change\s+to|change\s+it\s+for|change\s+for|change)\s+",
        r"^(?:move\s+it\s+to|move\s+to|move\s+it\s+for|move\s+for|move)\s+",
        r"^(?:postpone\s+it\s+to|postpone\s+to|postpone\s+it\s+for|postpone\s+for|postpone)\s+",
        r"^(?:how\s+about|what\s+about|make\s+it\s+for|make\s+it|keep\s+it\s+for|keep\s+it\s+at|keep\s+it|put\s+it\s+for|do\s+it\s+for)\s+",
        r"^(?:yes|yeah|yep|sure|okay|ok|haan|ha|हाँ|जी हाँ|అవును|సరే)[,\s]+",
        r"^(?:for|at|on|to)\s+",
        r"^(?:कृपया\s+इसे|कृपया)\s+",
        r"^(?:रीशेड्यूल\s+कर\s+दो|रीशेड्यूल\s+कर\s+दीजिए|रीशेड्यूल\s+कर\s+दें|रीशेड्यूल\s+करना\s+है|रीशेड्यूल)\s+",
        r"^(?:బహుశా|దయచేసి|రీషెడ్యూల్\s+చేయండి|రీషెడ్యూల్)\s+"
    ]

    changed = True
    while changed:
        changed = False
        for p in prefixes:
            new_cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
            if new_cleaned != cleaned and new_cleaned:
                cleaned = new_cleaned
                changed = True

    suffixes = [
        r"\s+(?:please|karo|kar do|kar dijiye|kariye|karein|hoga|rahega|karna hai|చేయండి|చెయ్యండి)$",
        r"\s+(?:रीशेड्यूल\s+कर\s+दो|रीशेड्यूल\s+कर\s+दीजिए|रीशेड्यूल\s+कर\s+दें|रीशेड्यूल\s+करना\s+है|रीशेड्यूल)$",
        r"\s+(?:రీషెడ్యూల్\s+చేయండి|రీషెడ్యూల్\s+చేయగలరా|రీషెడ్యూల్)$",
        r"\s+(?:would be fine|would be good|would work|works for me|is fine|is good)$"
    ]
    for s in suffixes:
        cleaned = re.sub(s, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = cleaned.rstrip(".,!?। ").strip()
    return cleaned if cleaned else None


def extract_customer_name_from_text(text: str, language: str = "en", agent_name: Optional[str] = None) -> Optional[str]:
    """
    Consolidated, language-aware customer name extractor.
    Supports English, Hindi, Hinglish, and Telugu patterns.

    agent_name: The current session's agent persona name (e.g. 'Maya', 'Arjun').
    Only THIS agent's name is blocked from being treated as the customer name.
    Other agent names are allowed — a real user may genuinely be named 'Arjun'
    while talking to agent 'Sophia'.
    """
    if not text:
        return None
        
    raw = text.strip().rstrip(".,!?।")
    
    INVALID_WORDS = {
        "unknown", "none", "null", "undefined", "n/a", "user", "customer",
        "my gosh", "in the car", "my car", "gosh", "yes", "no", "hello", "hi", "ok", "okay", "yep", "yeah", "sure",
        "go", "let's", "lets", "let", "come", "start", "see", "look", "show", "tell", "give", "speak", "speaking", "talk", "hear", "listen", "calling",
        "please", "today", "tomorrow", "yesterday", "appointment", "reschedule", "confirm", "cancel", "hospital", "doctor", "receptionist",
        "mera", "meri", "mere", "naam", "name", "hai", "hoon", "hu", "haan", "nahi", "bol", "raha", "rahi", "rahe", "baat", "kar", "kare", "ji", "ka", "ke", "ki", "ko", "se",
        "i", "i'm", "im", "my", "this", "it's", "it", "myself", "am", "called", "here", "hey", "from",
        "a", "an", "the", "and", "or", "is", "are", "me", "we", "you", "he", "she", "they", "sir", "madam", "maam", "mr", "mrs", "ms", "dr",
        "మీరు", "నా", "పేరు", "నమస్కారం", "అవును", "సరే", "ధన్యవాదాలు", "మాట్లాడుతున్నాను", "మాట్లాడుతున్నా", "నేను",
        "మరియు", "లేదా", "ఉంది", "ఉన్నారు"
    }
    # agent_name is intentionally NOT added to INVALID_WORDS.
    # Explicit pattern matches (steps 1-3) must always trust what the user said.
    # The agent name is only blocked in the single-word fallback (step 4) below.

    def clean_name(name_str: str) -> Optional[str]:
        if not name_str:
            return None
        words = name_str.strip().split()
        cleaned_words = []
        for w in words:
            w_clean = w.strip().rstrip(".,!?।")
            if w_clean.lower() not in INVALID_WORDS and len(w_clean) >= 2:
                cleaned_words.append(w_clean)
        if cleaned_words:
            return " ".join(w.title() for w in cleaned_words[:3])
        return None

    # 1. Regex patterns for English
    en_patterns = [
        r"\b(?:my name is|i am|i\'m|im|this is|myself|call me|you can call me)\s+([A-Za-z\s]+)",
        r"\b([A-Za-z\s]+)\s+(?:speaking|here|this side|calling)",
        r"\b(?:mera name|mera naam|main|naam|naam hai)\s+([A-Za-z\s]+)"
    ]
    
    for pat in en_patterns:
        match = re.search(pat, raw, re.IGNORECASE)
        if match:
            cand = clean_name(match.group(1))
            if cand:
                return cand

    # 2. Regex patterns for Hindi
    hi_patterns = [
        r"(?:मेरा नाम|नाम|मेरा नाम है)\s+([\u0900-\u097F\sA-Za-z]+)",
        r"(?:मैं|मै|हम)\s+([\u0900-\u097F\sA-Za-z]+)\s+(?:बोल\s+रहा|बोल\s+रही|बात\s+कर|बोल\s+रहे)",
        r"([\u0900-\u097F\sA-Za-z]+)\s+(?:बोल\s+रहा|बोल\s+रही|बात\s+कर|बोलता|बोलती)"
    ]
    for pat in hi_patterns:
        match = re.search(pat, raw, re.IGNORECASE)
        if match:
            cand_raw = match.group(1)
            for stop in ["है", "हूँ", "जी", "बात", "रहा", "रही", "बोल", "hoon", "hai"]:
                cand_raw = re.sub(rf"\s+{stop}\b", "", cand_raw, flags=re.IGNORECASE).strip()
            cand = clean_name(cand_raw)
            if cand:
                return cand

    # 3. Regex patterns for Telugu
    te_patterns = [
        r"(?:నా పేరు|పేరు)\s+([\u0C00-\u0C7F\sA-Za-z]+)",
        r"(?:నేను)\s+([\u0C00-\u0C7F\sA-Za-z]+)\s+(?:మాట్లాడుతున్నాను|మాట్లాడుతున్నా)",
        r"([\u0C00-\u0C7F\sA-Za-z]+)\s+(?:మాట్లాడుతున్నాను|మాట్లాడుతున్నా)"
    ]
    for pat in te_patterns:
        match = re.search(pat, raw)
        if match:
            cand_raw = match.group(1)
            for stop in ["మాట్లాడుతున్నాను", "మాట్లాడుతున్నా", "మాట్లాడు", "నేను"]:
                cand_raw = re.sub(rf"\s+{stop}\b", "", cand_raw).strip()
            cand = clean_name(cand_raw)
            if cand:
                return cand

    # 4. Fallback: single / double / triple word input with no surrounding context.
    #
    # The agent name is blocked HERE AND ONLY HERE.
    # A bare single word exactly matching the agent name is the signature of
    # greeting echo: VAD occasionally captures the tail of the agent's own
    # "Hi, this is Maya from CityCare..." audio, and Whisper transcribes just
    # "Maya". Blocking it here prevents a false customer_name assignment.
    #
    # This does NOT break the real-user case:
    #   "My name is Maya" (agent Maya) -> matched by step 1 above, never reaches here
    #   "Maya speaking"   (agent Maya) -> matched by step 1 above, never reaches here
    #   "Maya Sharma"     (agent Maya) -> 2 words, the guard below does NOT fire
    #   bare "Maya"       (agent Maya) -> 1 word == agent name -> correctly rejected
    words = raw.split()
    if 1 <= len(words) <= 3:
        if (len(words) == 1
                and agent_name
                and words[0].lower() == agent_name.strip().lower()):
            return None  # Single bare word == agent name -> likely greeting echo
        cleaned = clean_name(raw)
        if cleaned:
            if re.search(r'[A-Za-z\u0900-\u097F\u0C00-\u0C7F]', cleaned):
                return cleaned

    return None


def validate_tts_speech(text: str) -> bool:
    """Strict TTS input validation: checks for internal leaks (brackets, json, variables)."""
    if not text:
        return True
    if "[" in text or "]" in text or "{" in text or "}" in text:
        return False
    if "customer_name=" in text or "intent=" in text or "state=" in text or "tool=" in text:
        return False
    return True


HOSPITAL_KNOWLEDGE_TOPICS = [
    {
        "keywords": ["parking", "park", "car", "vehicle", "garage", "पार्किंग", "गाड़ी", "పార్కింగ్"],
        "answers": {
            "en": "Parking at CityCare Hospital is completely free for patients and visitors in our adjacent multi-story parking garage.",
            "hi": "सिटीकेयर हॉस्पिटल में सभी मरीजों और आगंतुकों के लिए मल्टी-स्टोरी पार्किंग गैरेज में पार्किंग पूरी तरह से मुफ्त है।",
            "te": "సిటీకేర్ హాస్పిటల్‌లో రోగులు మరియు సందర్శకులందరికీ మా మల్టీ-స్టోరీ పార్కింగ్ గ్యారేజీలో పార్కింగ్ పూర్తిగా ఉచితం."
        }
    },
    {
        "keywords": ["location", "located", "where", "address", "landmark", "directions", "reach", "metro", "कहाँ", "पता", "लोकेशन", "ఎక్కడ", "చిరునామా", "లొకేషన్"],
        "answers": {
            "en": "CityCare Hospital is located at 123 Health Ave, Suite 100, City Center, Mumbai, right near Central Park Metro Station.",
            "hi": "सिटीकेयर हॉस्पिटल 123 हेल्थ एवेन्यू, सिटी सेंटर, मुंबई में सेंट्रल पार्क मेट्रो स्टेशन के पास स्थित है।",
            "te": "సిటీకేర్ హాస్పిటల్ ముంబైలోని సిటీ సెంటర్, 123 హెల్త్ అవెన్యూ వద్ద సెంట్రల్ పార్క్ మెట్రో స్టేషన్ సమీపంలో ఉంది."
        }
    },
    {
        "keywords": ["timing", "timings", "hours", "open", "close", "working hours", "opd", "समय", "खुलने", "సమయం", "వేళలు"],
        "answers": {
            "en": "Our OPD working hours are from 8:00 AM to 8:00 PM daily, and our Emergency Room and ICU operate 24/7.",
            "hi": "हमारा ओपीडी रोजाना सुबह 8:00 बजे से रात 8:00 बजे तक खुला रहता है, और इमरजेंसी सेवा 24/7 खुली है।",
            "te": "మా OPD ప్రతిరోజూ ఉదయం 8:00 నుండి రాత్రి 8:00 వరకు తెరిచి ఉంటుంది, మరియు ఎమర్జెన్సీ 24/7 అందుబాటులో ఉంటుంది."
        }
    },
    {
        "keywords": ["fee", "fees", "cost", "charge", "charges", "price", "consultation", "फीस", "खर्चा", "शुल्क", "ఫీజు", "ఖర్చు"],
        "answers": {
            "en": "Consultation fee for Dr. Sharma is ₹800, Dr. Patel is ₹1000, and general OPD consultation is ₹500.",
            "hi": "डॉ. शर्मा की कंसल्टेशन फीस ₹800, डॉ. पटेल की ₹1000 और सामान्य ओपीडी फीस ₹500 है।",
            "te": "డాక్టర్ శర్మ కన్సల్టేషన్ ఫీజు ₹800, డాక్టర్ పటేల్ ఫీజు ₹1000 మరియు జనరల్ OPD ఫీజు ₹500."
        }
    },
    {
        "keywords": ["insurance", "cashless", "tpa", "claim", "mediclaim", "star health", "hdfc ergo", "icici lombard", "max bupa", "बीमा", "इन्श्योरेंस", "ఇన్సూరెన్స్"],
        "answers": {
            "en": "We support cashless insurance with major providers including Star Health, HDFC ERGO, ICICI Lombard, Max Bupa, Care Health, and Bajaj Allianz.",
            "hi": "हम स्टार हेल्थ, एचडीएफसी एर्गो, आईसीआईसीआई लोम्बार्ड, मैक्स बूपा और बजाज आलियांज सहित प्रमुख बीमा कंपनियों के लिए कैशलेस सुविधा प्रदान करते हैं।",
            "te": "మేము స్టార్ హెల్త్, HDFC ERGO, ICICI లాంబార్డ్ మరియు మ్యాక్స్ బూపాతో సహా ప్రధాన బీమా కంపెనీలకు క్యాష్‌లెస్ సౌకర్యాన్ని అందిస్తాము."
        }
    },
    {
        "keywords": ["emergency", "ambulance", "icu", "urgent", "casualty", "इमरजेंसी", "एंबुलेंस", "ఎమర్జెన్సీ", "అంబులెన్స్"],
        "answers": {
            "en": "CityCare Hospital has a 24/7 Emergency Room, ICU, and dedicated 24/7 ambulance support reachable at +91 22 5550 9999.",
            "hi": "सिटीकेयर हॉस्पिटल में 24/7 इमरजेंसी रूम, आईसीयू और 24/7 एम्बुलेंस सेवा (+91 22 5550 9999) उपलब्ध है।",
            "te": "సిటీకేర్ హాస్పిటల్‌లో 24/7 ఎమర్జెన్సీ రూమ్, ICU మరియు 24/7 అంబులెన్స్ సదుపాయం (+91 22 5550 9999) అందుబాటులో ఉన్నాయి."
        }
    },
    {
        "keywords": ["pharmacy", "medicine", "medicines", "medical shop", "chemist", "दवा", "दवाई", "दुकान", "మందులు", "ఫార్మసీ"],
        "answers": {
            "en": "Our in-house 24/7 pharmacy is located on the ground floor with valid prescription, and home delivery is available.",
            "hi": "हमारी 24/7 फार्मेसी ग्राउंड फ्लोर पर स्थित है और दवाओं की होम डिलीवरी भी उपलब्ध है।",
            "te": "మా 24/7 ఫార్మసీ గ్రౌండ్ ఫ్లోర్‌లో ఉంది మరియు హోమ్ డెలివరీ కూడా అందుబాటులో ఉంది."
        }
    },
    {
        "keywords": ["lab", "test", "blood test", "ecg", "diagnostic", "xray", "scan", "sample", "सैंपल", "जांच", "टेस्ट", "ల్యాబ్", "పరీక్షలు"],
        "answers": {
            "en": "Our in-house diagnostic laboratory operates from 7:00 AM to 9:00 PM daily with home sample collection available.",
            "hi": "हमारी डायग्नोस्टिक लैब रोजाना सुबह 7:00 बजे से रात 9:00 बजे तक खुली रहती है और होम सैंपल कलेक्शन उपलब्ध है।",
            "te": "మా డయాగ్నస్టిక్ ల్యాబ్ ప్రతిరోజూ ఉదయం 7:00 నుండి రాత్రి 9:00 వరకు పనిచేస్తుంది మరియు హోమ్ శాంపిల్ కలెక్షన్ అందుబాటులో ఉంది."
        }
    },
    {
        "keywords": ["canteen", "food", "cafeteria", "coffee", "tea", "meals", "breakfast", "lunch", "कैंटीन", "खाना", "కాంటీన్", "ఆహారం"],
        "answers": {
            "en": "Our hospital cafeteria is located on the ground floor, open daily from 7:00 AM to 10:00 PM.",
            "hi": "हमारा हॉस्पिटल कैफेटेरिया ग्राउंड फ्लोर पर स्थित है, जो सुबह 7:00 से रात 10:00 बजे तक खुला रहता है।",
            "te": "మా హాస్పిటల్ కెఫెటేరియా గ్రౌండ్ ఫ్లోర్‌లో ఉంది, ప్రతిరోజూ ఉదయం 7:00 నుండి రాత్రి 10:00 వరకు తెరిచి ఉంటుంది."
        }
    },
    {
        "keywords": ["cancel", "cancellation", "cancellation fee", "charges for cancel", "refund", "कैंसिल", "रद्द", "రద్దు"],
        "answers": {
            "en": "Appointments can be rescheduled or cancelled at least 24 hours in advance without any fee or cancellation charges.",
            "hi": "अपॉइंटमेंट को बिना किसी शुल्क के 24 घंटे पहले तक कैंसिल या रीशेड्यूल किया जा सकता है।",
            "te": "ఎలాంటి రుసుము లేకుండా అపాయింట్‌మెంట్‌ను 24 గంటల ముందు వరకు రద్దు చేసుకోవచ్చు లేదా రీషెడ్యూల్ చేయవచ్చు."
        }
    }
]


def get_deterministic_fallback(
    industry: str,
    state: str,
    lang_key: str,
    customer_name: Optional[str] = None,
    collected_info: Optional[dict] = None
) -> str:
    """Returns a completely safe, hardcoded response for the given state and language."""
    template_map = {
        "HOSPITAL_GREETING": "HOSPITAL_GREETING",
        "HOSPITAL_WAITING_FOR_NAME": "RECOVERY",
        "HOSPITAL_PURPOSE": "HOSPITAL_PURPOSE",
        "HOSPITAL_WAITING_FOR_DECISION": "UNCLEAR",
        "HOSPITAL_WAITING_FOR_RESCHEDULE_SLOT": "RESCHEDULE",
        "HOSPITAL_GOODBYE": "CLOSING",
        "RE_GREETING": "RE_GREETING",
        "RE_WAITING_FOR_NAME": "RECOVERY",
        "RE_INTEREST_DECISION": "RE_PURPOSE_INTRO",
        "RE_REQUIREMENT_COLLECTION": "RE_INTEREST_QUESTION",
        "RE_SITE_VISIT_OFFER": "RE_RECOMMENDATION",
        "RE_CALL_ENDED": "NOT_INTERESTED"
    }
    
    tpl_name = template_map.get(state, "RECOVERY")
    try:
        tpl_text = TEMPLATES[industry][tpl_name][lang_key]
    except KeyError:
        try:
            tpl_text = TEMPLATES[industry]["RECOVERY"][lang_key]
        except KeyError:
            tpl_text = "Hello, can you hear me?"
        
    res = tpl_text.format(
        agent_name="Sophia",
        customer_name=customer_name or "",
        slot=(collected_info or {}).get("reschedule_slot", "tomorrow"),
        unit_choice=(collected_info or {}).get("unit_choice", "premium 2 BHK apartment" if lang_key == "en" else "प्रीमियम 2 BHK फ्लैट")
    )
    res = res.replace("Hi ,", "Hi,").replace("Hello ,", "Hello,").replace("  ", " ").strip()
    return res


async def extract_speech_from_json_stream(token_stream) -> AsyncGenerator[str, None]:
    """Parses LLM token stream yielding structured JSON on the fly, and extracts ONLY the 'speech' string field."""
    buffer = ""
    in_speech_value = False
    escaped = False
    speech_found = False
    quote_char = None
    all_tokens = []
    yielded_any = False
    
    async for token, _ in token_stream:
        if token:
            all_tokens.append(token)
            buffer += token
            
            if not speech_found:
                speech_key_idx = buffer.find('"speech"')
                if speech_key_idx == -1:
                    speech_key_idx = buffer.find("'speech'")
                
                if speech_key_idx != -1:
                    sub_buf = buffer[speech_key_idx:]
                    colon_idx = sub_buf.find(':')
                    if colon_idx != -1:
                        quote_match = re.search(r'["\']', sub_buf[colon_idx:])
                        if quote_match:
                            quote_char = quote_match.group(0)
                            quote_pos = colon_idx + quote_match.start()
                            in_speech_value = True
                            speech_found = True
                            buffer = sub_buf[quote_pos + 1:]
            
            if in_speech_value:
                i = 0
                yield_buf = ""
                while i < len(buffer):
                    char = buffer[i]
                    if escaped:
                        yield_buf += char
                        escaped = False
                    elif char == '\\':
                        escaped = True
                    elif char == quote_char:
                        in_speech_value = False
                        buffer = buffer[i+1:]
                        break
                    else:
                        yield_buf += char
                    i += 1
                
                if yield_buf:
                    clean_chunk = clean_speech_text(yield_buf)
                    if clean_chunk:
                        yielded_any = True
                        yield clean_chunk
                
                if not in_speech_value:
                    buffer = ""
                else:
                    buffer = ""
                    
    if not yielded_any:
        full_raw = "".join(all_tokens).strip()
        if full_raw:
            if full_raw.startswith("{") and "speech" in full_raw:
                try:
                    data = json.loads(full_raw)
                    speech = data.get("speech", "")
                    if speech:
                        yield clean_speech_text(speech)
                        return
                except Exception:
                    pass
            yield clean_speech_text(full_raw)


def is_medical_safety_query(text: str) -> bool:
    """Detect if the user is describing personal symptoms or asking for medical diagnosis/treatment advice."""
    if not text:
        return False
    t_low = text.lower()

    # Direct emergency or personal symptom / diagnosis keywords (override question check)
    direct_safety_indicators = [
        "chest pain", "heart attack", "stroke", "shortness of breath", "severe pain",
        "diagnose", "diagnosis", "what disease", "what illness", "sick", "vomiting blood",
        "bleeding profusely", "unconscious"
    ]
    if any(ds in t_low for ds in direct_safety_indicators):
        return True

    # General symptom / medical keywords
    safety_keywords = [
        "pain", "fever", "headache", "disease", "symptom", "treatment", "vomiting",
        "bleeding", "dizzy", "cough", "infection", "bitten",
        "मर्ज", "बीमारी", "दर्द", "बुखार", "दवा", "इलाज", "लक्षण",
        "నొప్పి", "జ్వరం", "మందు", "చికిత్స", "లక్షణాలు"
    ]

    has_safety_kw = any(kw in t_low for kw in safety_keywords)
    if not has_safety_kw:
        return False

    # Hospital service / info questions that mention medical terms (e.g. "What department handles heart disease?")
    hospital_service_indicators = [
        "department", "clinic", "doctor", "dr.", "consultation", "fee", "cost", "timing",
        "hours", "located", "location", "address", "parking", "insurance", "appointment",
        "विभाग", "फीस", "समय", "पता"
    ]
    if any(hsi in t_low for hsi in hospital_service_indicators):
        return False

    return True


def is_doctor_query(text: str) -> bool:
    """Detect if user is asking who their doctor is or details about Dr. Sharma."""
    if not text:
        return False
    t_low = text.lower().strip()
    doc_patterns = [
        "which doctor", "who is the doctor", "who is my doctor", "what doctor",
        "toctr", "who am i seeing", "doctor name", "name of the doctor",
        "talk to", "appointment is with", "doctor's name", "dr sharma", "dr. sharma",
        "कौन से डॉक्टर", "डॉक्टर कौन हैं", "डॉक्टर का नाम",
        "ఏ డాక్టర్", "డాక్టర్ ఎవరు", "డాక్టర్ పేరు"
    ]
    return any(p in t_low for p in doc_patterns)


def is_appointment_time_query(text: str) -> bool:
    """Detect if the user is asking about the date/time/timing of their upcoming appointment."""
    if not text:
        return False
    t_low = text.lower().strip()
    patterns = [
        "what is the time", "what time", "timing of this appointment", "time of this appointment",
        "when is my appointment", "when is the appointment", "appointment time", "appointment timing",
        "which appointment i have", "which appointment", "what day", "what date", "appointment schedule",
        "समय क्या है", "अपॉइंटमेंट का समय", "कब है अपॉइंटमेंट", "कब है",
        "సమయం ఏమిటి", "అపాయింట్‌మెంట్ సమయం", "ఎప్పుడు"
    ]
    return any(p in t_low for p in patterns)


def is_other_doctors_query(text: str) -> bool:
    """Detect if the user is asking about other available doctors in the hospital."""
    if not text:
        return False
    t_low = text.lower().strip()
    patterns = [
        "other doctor", "other doctors", "another doctor", "different doctor",
        "other dr", "any other doctor", "who else", "which other doctor",
        "दूसरे डॉक्टर", "अन्य डॉक्टर", "कोई और डॉक्टर",
        "ఇతర డాక్టర్", "వేరే డాక్టర్", "మరేదైనా డాక్టర్"
    ]
    return any(p in t_low for p in patterns)


def resolve_hospital_direct_knowledge(user_text: str, lang_key: str) -> str | None:
    """Resolve direct hospital queries (doctor name, appointment time, parking, fees, location, etc.) directly without RAG."""
    if not user_text:
        return None
    t_low = user_text.lower().strip()
    
    if is_appointment_time_query(user_text):
        return {
            "en": "Your appointment with Dr. Sharma is scheduled for tomorrow at 11:00 AM.",
            "hi": "आपका डॉ. शर्मा के साथ अपॉइंटमेंट कल सुबह 11:00 बजे है।",
            "te": "డాక్టర్ శర్మగారితో మీ అపాయింట్‌మెంట్ రేపు ఉదయం 11:00 గంటలకు ఉంది."
        }.get(lang_key)
    elif is_other_doctors_query(user_text):
        return {
            "en": "Besides Dr. Sharma (Senior Cardiologist), CityCare Hospital also has Dr. Patel (Neurologist) and Dr. Mehta (Senior Pediatrician) available.",
            "hi": "डॉ. शर्मा के अलावा, सिटीकेयर हॉस्पिटल में डॉ. पटेल (न्यूरोलॉजिस्ट) और डॉ. मेहता (वरिष्ठ बाल रोग विशेषज्ञ) भी उपलब्ध हैं।",
            "te": "డాక్టర్ శర్మతో పాటు, సిటీకేర్ హాస్పిటల్‌లో డాక్టర్ పటేల్ (న్యూరాలజిస్ట్) మరియు డాక్టర్ మెహతా (సీనియర్ పీడియాట్రీషియన్) కూడా అందుబాటులో ఉన్నారు."
        }.get(lang_key)
    elif is_doctor_query(user_text):
        return {
            "en": "Your appointment is with Dr. Sharma, Senior Cardiologist and Heart Specialist at CityCare Hospital.",
            "hi": "आपका अपॉइंटमेंट सिटीकेयर हॉस्पिटल के वरिष्ठ कार्डियोलॉजिस्ट और हृदय रोग विशेषज्ञ डॉ. शर्मा के साथ है।",
            "te": "మీ అపాయింట్‌మెంట్ సిటీకేర్ హాస్పిటల్‌లోని సీనియర్ కార్డియాలజిస్ట్ మరియు హృద్రోగ నిపుణుడు డాక్టర్ శర్మగారితో ఉంది."
        }.get(lang_key)

    for topic in HOSPITAL_KNOWLEDGE_TOPICS:
        if any(re.search(r'\b' + re.escape(k) + r'\b', t_low) for k in topic["keywords"]):
            return topic["answers"].get(lang_key, topic["answers"]["en"])

    return None


def is_real_estate_knowledge_query(text: str) -> bool:
    """Detect if user is asking a real estate knowledge query (price, location, amenities, loan, possession, etc.)."""
    if not text:
        return False
    t_low = text.lower().strip()
    patterns = [
        "price", "cost", "how much", "rate", "budget", "pricing",
        "location", "where", "located", "address", "gachibowli", "landmark", "distance", "connectivity", "metro", "airport", "how far",
        "amenities", "facility", "facilities", "pool", "gym", "clubhouse", "parking", "security", "backup",
        "possession", "ready to move", "when will it", "completion", "handover",
        "loan", "bank", "finance", "emi", "interest",
        "rera", "approved", "approval", "builder", "skyline", "project", "developer", "developers", "legal",
        "sample flat", "model flat", "floor plan", "sqft", "square feet", "area", "size",
        "कीमत", "दाम", "कहाँ", "लोकेशन", "सुविधाएं", "कब्जा", "लोन", "बिल्डर", "रेरा",
        "ధర", "ఎక్కడ", "లొకేషన్", "సౌకర్యాలు", "పొసెషన్", "లోన్", "బిల్డర్"
    ]
    return any(p in t_low for p in patterns)


def resolve_real_estate_direct_knowledge(user_text: str, lang_key: str) -> str | None:
    """Resolve direct real estate queries (location, price, amenities, possession, loan, rera, builder) directly."""
    if not user_text:
        return None
    t_low = user_text.lower().strip()

    # Location & Connectivity query
    if any(p in t_low for p in ["location", "where", "located", "address", "gachibowli", "metro", "airport", "distance", "connectivity", "how far", "लोकेशन", "कहाँ", "दूरी", "ఎక్కడ", "లొకేషన్", "దూరం"]):
        return {
            "en": "Skyline Residency is located in prime Gachibowli, Hyderabad, just 5 minutes from the Financial District, 10 minutes from Hitec City, and 25 minutes from the Airport.",
            "hi": "Skyline Residency हैदराबाद के गचीबोवली में स्थित है, जो फाइनेंशियल डिस्ट्रिक्ट से 5 मिनट और एयरपोर्ट से 25 मिनट की दूरी पर है।",
            "te": "Skyline Residency హైదరాబాద్‌లోని గచ్చిబౌలిలో ఉంది, ఫైనాన్షియల్ డిస్ట్రిక్ట్ నుండి 5 నిమిషాలు మరియు ఎయిర్‌పోర్ట్ నుండి 25 నిమిషాల దూరం."
        }.get(lang_key)

    # Price / cost query
    if any(p in t_low for p in ["price", "cost", "how much", "rate", "budget", "pricing", "कीमत", "दाम", "कितना", "ధర", "ఎంత"]):
        return {
            "en": "Our 2 BHK starts at 80 Lakhs, 3 BHK at 1.2 Crores, and luxury duplex Penthouses at 2.5 Crores, with flexible payment plans.",
            "hi": "हमारे 2 BHK की कीमत 80 लाख, 3 BHK 1.2 करोड़ और डुप्लेक्स पेंटहाउस 2.5 करोड़ रुपये से शुरू है।",
            "te": "మా 2 BHK 80 లక్షలు, 3 BHK 1.2 కోట్లు మరియు డ్యూప్లెక్స్ పెంట్‌హౌస్ 2.5 కోట్ల నుండి ప్రారంభమవుతాయి."
        }.get(lang_key)

    # Amenities query
    if any(p in t_low for p in ["amenities", "facility", "facilities", "pool", "gym", "clubhouse", "parking", "security", "सुविधाएं", "पूल", "जिम", "సౌకర్యాలు"]):
        return {
            "en": "Skyline Residency offers a 20,000 sq ft clubhouse, swimming pool, fully equipped gym, children's play area, covered parking, and 24/7 security.",
            "hi": "Skyline Residency में 20,000 वर्ग फुट का क्लब हाउस, स्विमिंग पूल, जिम, बच्चों का पार्क और 24/7 सुरक्षा उपलब्ध है।",
            "te": "Skyline Residency లో 20,000 చదరపు అడుగుల క్లబ్‌హౌస్, స్విమ్మింగ్ పూల్, జిమ్ మరియు 24/7 భద్రత అందుబాటులో ఉన్నాయి."
        }.get(lang_key)

    # Possession query
    if any(p in t_low for p in ["possession", "ready to move", "completion", "handover", "when", "date", "status", "कब्जा", "कब", "పొసెషన్"]):
        return {
            "en": "The project is under active construction with handovers starting in December 2026. Fully furnished sample flats are ready for site visits now.",
            "hi": "प्रोजेक्ट का निर्माण कार्य जारी है और पजेशन दिसंबर 2026 से शुरू होगा। सैंपल फ्लैट्स देखने के लिए तैयार हैं।",
            "te": "ప్రాజెక్ట్ నిర్మాణం జరుగుతోంది, డిసెంబర్ 2026 నుండి హ్యాండోవర్ ప్రారంభమవుతుంది. శాంపిల్ ఫ్లాట్లు ఇప్పుడు సిద్ధంగా ఉన్నాయి."
        }.get(lang_key)

    # Loan query
    if any(p in t_low for p in ["loan", "bank", "finance", "emi", "interest", "लोन", "लोन सुविधा", "లోన్"]):
        return {
            "en": "We have pre-approved home loan partnerships with SBI, HDFC, ICICI, and Axis Bank with competitive interest rates and easy EMI options.",
            "hi": "हमारे पास एसबीआई, एचडीएफसी, आईसीआईसीआई और एक्सिस बैंक के साथ प्री-अप्रूव्ड होम लोन और आसान ईएमआई सुविधाएं उपलब्ध हैं।",
            "te": "మా వద్ద SBI, HDFC, ICICI మరియు Axis బ్యాంక్‌ల నుండి ప్రీ-అప్రూవ్డ్ హోమ్ లోన్ మరియు EMI సౌకర్యం ఉంది."
        }.get(lang_key)

    # RERA & Approvals query
    if any(p in t_low for p in ["rera", "approved", "approval", "registration", "legal", "permissions", "रेरा", "అనుమతులు"]):
        return {
            "en": "Skyline Residency is fully RERA approved and GHMC certified with 100% clear legal titles and environmental clearances.",
            "hi": "Skyline Residency पूरी तरह से RERA और GHMC द्वारा स्वीकृत है और सभी कानूनी मंजूरियां प्राप्त हैं।",
            "te": "Skyline Residency పూర్తిగా RERA మరియు GHMC అనుమతులు పొందిన ప్రాజెక్ట్."
        }.get(lang_key)

    # Developer & Builder query
    if any(p in t_low for p in ["builder", "developer", "developers", "company", "who is building", "बिल्डर", "कंपनी", "బిల్డర్"]):
        return {
            "en": "Skyline Developers is one of Hyderabad's premier real estate builders with over 15 years of excellence and 12 successfully delivered projects.",
            "hi": "Skyline Developers हैदराबाद के प्रतिष्ठित बिल्डर्स में से एक हैं, जिन्होंने 15 वर्षों में 12 से अधिक सफल प्रोजेक्ट्स डिलीवर किए हैं।",
            "te": "Skyline Developers హైదరాబాద్‌లో 15 సంవత్సరాల అనుభవంతో 12 కంటే ఎక్కువ విజయవంతమైన ప్రాజెక్ట్‌లను పూర్తి చేసిన సంస్థ."
        }.get(lang_key)

    return None


def is_hospital_knowledge_query(text: str) -> bool:
    """Detect if the user is asking a hospital knowledge or policy question."""
    if not text:
        return False
    t_low = text.lower().strip()
    
    if is_doctor_query(text) or is_appointment_time_query(text) or is_other_doctors_query(text):
        return True

    # Exclude explicit pure commands
    if t_low in ["confirm", "confirm it", "confirm my appointment", "cancel", "cancel it", "cancel my appointment", "reschedule", "reschedule it"]:
        return False

    query_patterns = [
        "what", "where", "when", "how much", "how many", "how", "why", "who", "which",
        "do you", "does", "can i", "can you", "could", "would", "should",
        "is there", "are there", "tell me", "know", "information", "about",
        "price", "fee", "cost", "charge", "timing", "hours", "open", "close",
        "address", "located", "location", "landmark", "available", "availability",
        "facility", "facilities", "department", "departments", "service", "services",
        "offer", "provide", "accept",
        "specialize", "specialty", "specialist", "specialization", "doctor", "dr.", "sharma",
        "patel", "mehta", "cardiologist", "orthopedic", "pediatrician", "ecg", "lab", "test",
        "pharmacy", "medicines", "insurance", "cashless", "policy", "happens if", "cancellation fee",
        "cancel fee", "reschedule fee", "parking", "emergency", "ambulance", "cafeteria",
        "visiting", "procedure", "process", "toctr",
        "कहाँ", "कब", "कितना", "कैसे", "क्यों", "फीस", "समय", "डॉक्टर", "पार्किंग", "बीमा", "इलाज", "सुविधा",
        "ఎక్కడ", "ఎప్పుడు", "ఎంత", "ఎలా", "ఎందుకు", "ఫీజు", "సమయం", "డాక్టర్", "పార్కింగ్", "ఇన్సూరెన్స్", "సేవలు"
    ]
    
    return any(pat in t_low for pat in query_patterns)
 

def validate_tool_call(
    industry: str,
    state: str,
    detected_intent: str,
    requested_tool: str,
    slots: Optional[dict] = None
) -> bool:
    allowed = False
    if industry == "hospital":
        if state in ("HOSPITAL_WAITING_FOR_DECISION", "HOSPITAL_POST_ACTION"):
            if detected_intent == "CONFIRM_APPOINTMENT" and requested_tool == "confirm_appointment":
                allowed = True
            elif detected_intent == "CANCEL_APPOINTMENT" and requested_tool == "cancel_appointment":
                allowed = True
            elif detected_intent == "RESCHEDULE_APPOINTMENT" and requested_tool == "reschedule_appointment":
                allowed = True
        elif state == "HOSPITAL_WAITING_FOR_RESCHEDULE_SLOT":
            if requested_tool == "reschedule_appointment":
                allowed = True
    elif industry == "real_estate":
        if state in ("RE_SITE_VISIT_OFFER", "RE_INTEREST_DECISION"):
            if detected_intent == "CONFIRM_SITE_VISIT" and requested_tool == "book_site_visit":
                allowed = True

    logger.info(
        f"[TOOL-GUARD]\n"
        f"industry={industry}\n"
        f"state={state}\n"
        f"intent={detected_intent}\n"
        f"requested_tool={requested_tool}\n"
        f"allowed={str(allowed).lower()}\n"
        f"slots={slots}\n"
        f"allowed_statuses={['Success'] if allowed else ['Failed: tool_not_allowed_in_current_state']}"
    )
    return allowed


def check_and_reject_tool_calls(industry: str, state: str, requested_tool: str) -> None:
    forbidden = ["HOSPITAL_GREETING", "HOSPITAL_WAITING_FOR_NAME", "HOSPITAL_NAME_CAPTURED", "HOSPITAL_PURPOSE"]
    if industry == "hospital" and state in forbidden:
        logger.warning(
            f"[TOOL-GUARD]\n"
            f"industry={industry}\n"
            f"state={state}\n"
            f"intent=FORBIDDEN_STATE\n"
            f"requested_tool={requested_tool}\n"
            f"allowed=false\n"
            f"slots=None\n"
            f"allowed_statuses=['Failed: tool_not_allowed_in_current_state']"
        )


def validate_response_against_state(
    response: str,
    state: str,
    industry: str,
    customer_name: Optional[str] = None
) -> Tuple[bool, str]:
    # Placeholder validation intercept
    return True, response


# ── EXTENDED METHODS FOR CLASS ───────────────────────────────────────────────
# We add process_voice_demo_turn_stream to the class dynamically by matching the target signature:

async def _process_voice_demo_turn_stream_impl(
    self,
    call_id: str,
    campaign_id: uuid.UUID,
    *args,
    **kwargs
) -> AsyncGenerator[Tuple[Optional[str], bool, bool], None]:
    """Dedicated voice demo turn stream with strict state machine templates and JSON parsing."""
    industry = kwargs.get("industry", "hospital")
    language = kwargs.get("language", "English")
    agent_name = kwargs.get("agent_name", "Sophia")
    user_text = kwargs.get("user_text", "")

    if args:
        if len(args) == 4:
            industry, language, agent_name, user_text = args

    meta = await self.session_manager.get_session_metadata(call_id)
    if not meta:
        meta = {"industry": industry, "language": language, "agent_name": agent_name}
        await self.session_manager.update_session_metadata(call_id, meta)

    agent_name = meta.get("agent_name", agent_name)
    lang_input = meta.get("language", "en").lower().strip()
    if lang_input in ("english", "en"):
        lang_code = "en"
        lang_str = "English"
    elif lang_input in ("hindi", "hi"):
        lang_code = "hi"
        lang_str = "Hindi"
    elif lang_input in ("telugu", "te"):
        lang_code = "te"
        lang_str = "Telugu"
    else:
        lang_code = "en"
        lang_str = "English"
    lang_key = lang_code
    industry = meta.get("industry", industry)
    current_state = await self.session_manager.get_session_state(call_id) or "CALL_STARTED"
    if current_state == "WAIT_FOR_NAME":
        if industry == "hospital":
            current_state = "HOSPITAL_WAITING_FOR_NAME"
        elif industry == "real_estate":
            current_state = "RE_WAITING_FOR_NAME"
    
    from app.services.session_manager import VoiceSession
    session_store = VoiceSession(call_id)
    customer_name = session_store.customer_name
    collected_info = meta.get("collected_info", {})

    history = await self.session_manager.get_message_history(call_id)
    assistant_msgs = [m for m in history if m["role"] == "assistant"]
    last_agent_message = assistant_msgs[-1]["content"] if assistant_msgs else ""

    should_hangup = False
    should_transfer = False
    tool_executed = None
    tool_result = None

    if current_state in ("HOSPITAL_GOODBYE", "HOSPITAL_CALL_ENDED", "RE_CALL_ENDED", "CALL_ENDED", "CALL_COMPLETED") or meta.get("should_hangup"):
        logger.info(f"[CONV-ENGINE] Session {call_id} is in terminal state. Ending.")
        yield None, True, False
        return

    target_template_name = None
    detected_intent = "UNKNOWN"
    allowed_intents = []
    validated_intent = "UNKNOWN"
    tool_allowed = False
    requested_tool = None
    next_state = current_state

    if industry == "hospital":
        if user_text == "[CALL_START]":
            current_state = "HOSPITAL_GREETING"
            target_template_name = "HOSPITAL_GREETING"
            next_state = "HOSPITAL_WAITING_FOR_NAME"
            detected_intent = "CALL_START"
            validated_intent = "CALL_START"
            allowed_intents = ["CALL_START"]
            await self.session_manager.update_session_state(call_id, next_state)

        elif current_state == "HOSPITAL_WAITING_FOR_NAME":
            allowed_intents = ["NAME", "UNKNOWN"]
            
            if not customer_name:
                extracted = extract_customer_name_from_text(user_text, lang_code, agent_name=agent_name)
                if extracted:
                    customer_name = extracted
                    session_store.customer_name = extracted
                    meta["customer_name"] = customer_name
                    collected_info["customer_name"] = customer_name
                    await self.session_manager.update_session_metadata(
                        call_id, {"customer_name": customer_name, "collected_info": collected_info}
                    )

            if customer_name:
                detected_intent = "NAME"
                validated_intent = "NAME"
                target_template_name = "HOSPITAL_PURPOSE"
                next_state = "HOSPITAL_WAITING_FOR_DECISION"
            else:
                detected_intent = "UNKNOWN"
                validated_intent = "UNKNOWN"
                target_template_name = "RECOVERY"
                next_state = "HOSPITAL_WAITING_FOR_NAME"

            await self.session_manager.update_session_state(call_id, next_state)

        elif current_state == "HOSPITAL_WAITING_FOR_DECISION":
            allowed_intents = ["CONFIRM_APPOINTMENT", "CANCEL_APPOINTMENT", "RESCHEDULE_APPOINTMENT", "KNOWLEDGE_QUERY", "MEDICAL_SAFETY_QUERY", "REPEAT", "AMBIGUOUS_YES"]
            
            t_lower = user_text.lower().strip().rstrip(".,!?।")
            confirm_words = ["confirm", "confirm it", "confirm my appointment", "keep it", "yes confirm", "kya confirm", "कन्फर्म", "कन्फर्म करें", "कन्फर्म कर दीजिए", "కన్ఫర్మ్", "కన్ఫర్మ్ చేయండి"]
            cancel_words = ["cancel", "cancel it", "cancel my appointment", "i want to cancel", "कैंसिल", "कैंसिल करें", "कंसल", "रद्द", "రద్దు చేయండి"]
            reschedule_intent_words = [
                "reschedule", "reshedule", "postpone", "change my appointment", "change the date",
                "change the time", "move my appointment", "risk a duel", "rescind", "reshuffle",
                "re schedule", "re-schedule", "receipt", "change date", "change time", "another day",
                "different day", "different time", "another time", "resk schedule", "resk", "recided",
                "appoint pent", "risk schedule", "riske", "रीशेड्यूल", "बदल", "రీషెడ్యూల్"
            ]
            repeat_words = ["repeat", "say again", "pardon", "what did you say", "dubara", "phir se", "fir se", "kya bola", "दोहराएं", "మరోసారి", "మళ్లీ"]
            ambiguous_yes_words = ["yes", "yeah", "yep", "sure", "okay", "ok", "haan", "ha", "हाँ", "जी हाँ", "అవును", "సరే"]
            
            # Policy/info keywords — when present alongside action words, treat as KNOWLEDGE query
            policy_keywords = ["fee", "cost", "charge", "policy", "happens if", "what if", "how much", "how does", "what about", "price", "penalty", "refund", "process", "procedure"]
            
            is_repeat = any(w in t_lower for w in repeat_words)
            is_safety = is_medical_safety_query(user_text)
            is_knowledge = is_hospital_knowledge_query(user_text)
            is_policy_q = any(pk in t_lower for pk in policy_keywords)
            
            has_confirm = any(w in t_lower for w in confirm_words)
            has_cancel = any(w in t_lower for w in cancel_words)
            has_reschedule = (
                any(w in t_lower for w in reschedule_intent_words)
                or ("resk" in t_lower and "schedule" in t_lower)
                or ("appoint" in t_lower and ("resk" in t_lower or "recid" in t_lower or "ris" in t_lower or "change" in t_lower))
                or ("wine plant" in t_lower or "riske" in t_lower or "recided" in t_lower)
            )
            has_action = has_confirm or has_cancel or has_reschedule
            
            if is_repeat:
                detected_intent = "REPEAT"
                validated_intent = "REPEAT"
                target_template_name = "REPEAT"
                next_state = current_state
            elif has_action and is_policy_q:
                # "What is the cancellation fee?" — action word present but it's a policy question
                detected_intent = "KNOWLEDGE_QUERY"
                validated_intent = "KNOWLEDGE_QUERY"
                next_state = current_state
            elif has_action:
                # Explicit action command: "cancel", "I want to reschedule", "confirm it"
                if has_reschedule:
                    detected_intent = "RESCHEDULE_APPOINTMENT"
                    validated_intent = "RESCHEDULE_APPOINTMENT"
                    requested_tool = "reschedule_appointment"
                elif has_cancel:
                    detected_intent = "CANCEL_APPOINTMENT"
                    validated_intent = "CANCEL_APPOINTMENT"
                    requested_tool = "cancel_appointment"
                elif has_confirm:
                    detected_intent = "CONFIRM_APPOINTMENT"
                    validated_intent = "CONFIRM_APPOINTMENT"
                    requested_tool = "confirm_appointment"
            elif is_safety:
                # Medical symptom description / safety query (takes priority over knowledge query)
                detected_intent = "MEDICAL_SAFETY_QUERY"
                validated_intent = "MEDICAL_SAFETY_QUERY"
                next_state = current_state
            elif is_knowledge:
                # Hospital knowledge question without action words
                detected_intent = "KNOWLEDGE_QUERY"
                validated_intent = "KNOWLEDGE_QUERY"
                next_state = current_state
            elif any(w == t_lower for w in ambiguous_yes_words):
                detected_intent = "AMBIGUOUS_YES"
                validated_intent = "AMBIGUOUS_YES"
                target_template_name = "AMBIGUOUS_YES"
                next_state = current_state
            else:
                detected_intent = "UNKNOWN"
                validated_intent = "UNCLEAR"
                target_template_name = "UNCLEAR"
                next_state = current_state

            if validated_intent == "MEDICAL_SAFETY_QUERY":
                logger.info(f"[HOSPITAL-INTENT] session={call_id} intent=MEDICAL_SAFETY_QUERY state={current_state}")
                safety_ans = TEMPLATES["hospital"]["MEDICAL_SAFETY"][lang_key]
                pending_prompt = {
                    "en": "\n\nWould you like to confirm, cancel, or reschedule your appointment?",
                    "hi": "\n\nक्या आप अपना अपॉइंटमेंट कन्फर्म, कैंसिल या रीशेड्यूल करना चाहेंगे?",
                    "te": "\n\nమీరు మీ అపాయింట్‌మెంట్‌ను కన్ఫర్మ్, క్యాన్సిల్ లేదా రీషెడ్యూల్ చేయాలనుకుంటున్నారా?"
                }[lang_key]
                response_text = safety_ans + pending_prompt
                yield response_text, False, False
                bot_turn = {"role": "assistant", "content": response_text}
                await self.session_manager.append_message(call_id, bot_turn)
                await self.session_manager.update_session_state(call_id, next_state)
                return

            elif validated_intent == "KNOWLEDGE_QUERY":
                logger.info(f"[HOSPITAL-RAG] session={call_id} query='{user_text}' state={current_state}")
                pending_prompt = {
                    "en": "\n\nWould you like to confirm, cancel, or reschedule your appointment?",
                    "hi": "\n\nक्या आप अपना अपॉइंटमेंट कन्फर्म, कैंसिल या रीशेड्यूल करना चाहेंगे?",
                    "te": "\n\nమీరు మీ అపాయింట్‌మెంట్‌ను కన్ఫర్మ్, క్యాన్సిల్ లేదా రీషెడ్యూల్ చేయాలనుకుంటున్నారా?"
                }[lang_key]

                direct_ans = resolve_hospital_direct_knowledge(user_text, lang_key)
                if direct_ans:
                    logger.info(f"[HOSPITAL-DIRECT-KNOWLEDGE] session={call_id} answer='{direct_ans}'")
                    response_text = direct_ans + pending_prompt
                else:
                    facts = await self.rag_service.search_knowledge(campaign_id, user_text, limit=2)
                    if facts and facts[0].get("score", 0) >= 0.15:
                        fact_text = facts[0]["text"]
                        logger.info(f"[HOSPITAL-RAG-RESULT] session={call_id} retrieved_fact='{fact_text}' score={facts[0].get('score'):.2f}")
                        response_text = fact_text + pending_prompt
                    else:
                        logger.info(f"[HOSPITAL-RAG-RESULT] session={call_id} no local facts matched, invoking Gemini LLM dynamically for query='{user_text}'.")
                        llm_prompt = f"The customer asked: '{user_text}'. Answer their question accurately, helpfully, and naturally in 1-2 sentences as the receptionist, then ask if they would like to confirm, cancel, or reschedule their appointment."
                        llm_resp, _ = await self.llm_service.generate_completion(history + [{"role": "user", "content": llm_prompt}])
                        if llm_resp:
                            response_text = clean_speech_text(llm_resp)
                        else:
                            response_text = TEMPLATES["hospital"]["UNKNOWN"][lang_key] + pending_prompt
                
                yield response_text, False, False
                bot_turn = {"role": "assistant", "content": response_text}
                await self.session_manager.append_message(call_id, bot_turn)
                await self.session_manager.update_session_state(call_id, next_state)
                return

            elif validated_intent == "CONFIRM_APPOINTMENT":
                tool_allowed = validate_tool_call(industry, current_state, validated_intent, requested_tool)
                logger.info(f"[HOSPITAL-TOOL-GUARD] session={call_id} tool=confirm_appointment allowed={tool_allowed}")
                if tool_allowed:
                    tool_executed = "confirm_appointment"
                    tool_result = "Success: Appointment confirmed."
                    logger.info(f"[HOSPITAL-ACTION] session={call_id} action=CONFIRM_APPOINTMENT result={tool_result}")
                    target_template_name = "CONFIRM"
                    next_state = "HOSPITAL_POST_ACTION"

            elif validated_intent == "CANCEL_APPOINTMENT":
                tool_allowed = validate_tool_call(industry, current_state, validated_intent, requested_tool)
                logger.info(f"[HOSPITAL-TOOL-GUARD] session={call_id} tool=cancel_appointment allowed={tool_allowed}")
                if tool_allowed:
                    tool_executed = "cancel_appointment"
                    tool_result = "Success: Appointment cancelled."
                    logger.info(f"[HOSPITAL-ACTION] session={call_id} action=CANCEL_APPOINTMENT result={tool_result}")
                    target_template_name = "CANCEL"
                    next_state = "HOSPITAL_POST_ACTION"

            elif validated_intent == "RESCHEDULE_APPOINTMENT":
                cleaned_slot = clean_reschedule_slot(user_text)
                if cleaned_slot:
                    tool_allowed = validate_tool_call(industry, current_state, validated_intent, requested_tool, slots={"slot": cleaned_slot})
                    logger.info(f"[HOSPITAL-TOOL-GUARD] session={call_id} tool=reschedule_appointment allowed={tool_allowed}")
                    if tool_allowed:
                        tool_executed = "reschedule_appointment"
                        tool_result = f"Success: Rescheduled to {cleaned_slot}"
                        collected_info["reschedule_slot"] = cleaned_slot
                        logger.info(f"[HOSPITAL-ACTION] session={call_id} action=RESCHEDULE_APPOINTMENT result={tool_result}")
                        target_template_name = "RESCHEDULE_CONFIRM"
                        next_state = "HOSPITAL_POST_ACTION"
                else:
                    target_template_name = "RESCHEDULE"
                    next_state = "HOSPITAL_WAITING_FOR_RESCHEDULE_SLOT"

            await self.session_manager.update_session_state(call_id, next_state)

        elif current_state == "HOSPITAL_WAITING_FOR_RESCHEDULE_SLOT":
            t_lower = user_text.lower().strip()
            is_safety = is_medical_safety_query(user_text)
            is_knowledge = is_hospital_knowledge_query(user_text)
            
            pending_prompt = {
                "en": "\n\nWhat day or time would you prefer for your rescheduled appointment?",
                "hi": "\n\nआप अपने रीशेड्यूल किए गए अपॉइंटमेंट के लिए कौन सा दिन या समय पसंद करेंगे?",
                "te": "\n\nమీ రీషెడ్యూల్డ్ అపాయింట్‌మెంట్ కోసం ఏ రోజు లేదా సమయం అనుకూలంగా ఉంటుంది?"
            }[lang_key]

            if is_safety:
                logger.info(f"[HOSPITAL-INTENT] session={call_id} intent=MEDICAL_SAFETY_QUERY state={current_state}")
                safety_ans = TEMPLATES["hospital"]["MEDICAL_SAFETY"][lang_key]
                response_text = safety_ans + pending_prompt
                yield response_text, False, False
                bot_turn = {"role": "assistant", "content": response_text}
                await self.session_manager.append_message(call_id, bot_turn)
                return
            elif is_knowledge:
                logger.info(f"[HOSPITAL-RAG] session={call_id} query='{user_text}' state={current_state}")
                direct_ans = resolve_hospital_direct_knowledge(user_text, lang_key)
                if direct_ans:
                    logger.info(f"[HOSPITAL-DIRECT-KNOWLEDGE] session={call_id} answer='{direct_ans}'")
                    response_text = direct_ans + pending_prompt
                else:
                    facts = await self.rag_service.search_knowledge(campaign_id, user_text, limit=2)
                    if facts and facts[0].get("score", 0) >= 0.15:
                        fact_text = facts[0]["text"]
                        logger.info(f"[HOSPITAL-RAG-RESULT] session={call_id} retrieved_fact='{fact_text}' score={facts[0].get('score'):.2f}")
                        response_text = fact_text + pending_prompt
                    else:
                        logger.info(f"[HOSPITAL-RAG-RESULT] session={call_id} no facts matched, using fallback.")
                        fallback_text = TEMPLATES["hospital"]["UNKNOWN"][lang_key]
                        response_text = fallback_text + pending_prompt
                yield response_text, False, False
                bot_turn = {"role": "assistant", "content": response_text}
                await self.session_manager.append_message(call_id, bot_turn)
                return
            else:
                cleaned_slot = clean_reschedule_slot(user_text)
                if cleaned_slot:
                    allowed_intents = ["PROVIDE_SLOT"]
                    detected_intent = "PROVIDE_SLOT"
                    validated_intent = "PROVIDE_SLOT"
                    requested_tool = "reschedule_appointment"
                    
                    collected_info["reschedule_slot"] = cleaned_slot
                    await self.session_manager.update_session_metadata(call_id, {"collected_info": collected_info})

                    tool_allowed = validate_tool_call(industry, current_state, validated_intent, requested_tool, slots={"slot": cleaned_slot})
                    logger.info(f"[HOSPITAL-TOOL-GUARD] session={call_id} tool=reschedule_appointment allowed={tool_allowed}")
                    if tool_allowed:
                        tool_executed = "reschedule_appointment"
                        tool_result = f"Success: Rescheduled to {cleaned_slot}"
                        logger.info(f"[HOSPITAL-ACTION] session={call_id} action=RESCHEDULE_APPOINTMENT result={tool_result}")
                        target_template_name = "RESCHEDULE_CONFIRM"
                        next_state = "HOSPITAL_POST_ACTION"
                else:
                    detected_intent = "RESCHEDULE_APPOINTMENT"
                    validated_intent = "RESCHEDULE_APPOINTMENT"
                    target_template_name = "RESCHEDULE"
                    next_state = "HOSPITAL_WAITING_FOR_RESCHEDULE_SLOT"

                await self.session_manager.update_session_state(call_id, next_state)

        elif current_state == "HOSPITAL_POST_ACTION":
            t_lower = user_text.lower().strip().rstrip(".,!?।")
            decline_words = [
                "no", "nope", "no thanks", "thanks", "thank you", "thanks a lot", "thank you so much",
                "nothing else", "that's all", "thats all", "all good", "bye", "goodbye", "nahi",
                "धन्यवाद", "शुक्रिया", "नहीं", "వద్దు", "లేదు", "సెలవు", "ధన్యవాదాలు"
            ]
            is_decline = any(w in t_lower for w in decline_words) or t_lower in decline_words
            is_safety = is_medical_safety_query(user_text)
            is_knowledge = is_hospital_knowledge_query(user_text)
            
            pending_prompt = {
                "en": "\n\nIs there anything else I can help you with?",
                "hi": "\n\nक्या मैं आपकी किसी और चीज़ में मदद कर सकती हूँ?",
                "te": "\n\nనేను మీకు ఇంకేమైనా సహాయం చేయగలనా?"
            }[lang_key]

            if is_decline:
                logger.info(f"[HOSPITAL-GOODBYE] session={call_id} user declined further help, sending goodbye.")
                detected_intent = "GOODBYE"
                validated_intent = "GOODBYE"
                target_template_name = "GOODBYE"
                next_state = "HOSPITAL_GOODBYE"
                should_hangup = True
            elif is_safety:
                logger.info(f"[HOSPITAL-INTENT] session={call_id} intent=MEDICAL_SAFETY_QUERY state={current_state}")
                safety_ans = TEMPLATES["hospital"]["MEDICAL_SAFETY"][lang_key]
                response_text = safety_ans + pending_prompt
                yield response_text, False, False
                bot_turn = {"role": "assistant", "content": response_text}
                await self.session_manager.append_message(call_id, bot_turn)
                return
            elif is_knowledge:
                logger.info(f"[HOSPITAL-RAG] session={call_id} query='{user_text}' state={current_state}")
                direct_ans = resolve_hospital_direct_knowledge(user_text, lang_key)
                if direct_ans:
                    logger.info(f"[HOSPITAL-DIRECT-KNOWLEDGE] session={call_id} answer='{direct_ans}'")
                    response_text = direct_ans + pending_prompt
                else:
                    facts = await self.rag_service.search_knowledge(campaign_id, user_text, limit=2)
                    if facts and facts[0].get("score", 0) >= 0.15:
                        fact_text = facts[0]["text"]
                        logger.info(f"[HOSPITAL-RAG-RESULT] session={call_id} retrieved_fact='{fact_text}' score={facts[0].get('score'):.2f}")
                        response_text = fact_text + pending_prompt
                    else:
                        logger.info(f"[HOSPITAL-RAG-RESULT] session={call_id} no local facts matched, invoking Gemini LLM dynamically for query='{user_text}'.")
                        llm_prompt = f"The customer asked: '{user_text}'. Answer their question accurately, helpfully, and naturally in 1-2 sentences as the receptionist."
                        llm_resp, _ = await self.llm_service.generate_completion(history + [{"role": "user", "content": llm_prompt}])
                        if llm_resp:
                            response_text = clean_speech_text(llm_resp)
                        else:
                            response_text = TEMPLATES["hospital"]["UNKNOWN"][lang_key] + pending_prompt
                yield response_text, False, False
                bot_turn = {"role": "assistant", "content": response_text}
                await self.session_manager.append_message(call_id, bot_turn)
                return
            else:
                target_template_name = "ANYTHING_ELSE"
                next_state = "HOSPITAL_POST_ACTION"

            await self.session_manager.update_session_state(call_id, next_state)

        if requested_tool and not tool_allowed:
            check_and_reject_tool_calls(industry, current_state, requested_tool)

    elif industry == "real_estate":
        if user_text == "[CALL_START]":
            current_state = "RE_GREETING"
            target_template_name = "RE_GREETING"
            next_state = "RE_WAITING_FOR_NAME"
            detected_intent = "CALL_START"
            validated_intent = "CALL_START"
            allowed_intents = ["CALL_START"]
            await self.session_manager.update_session_state(call_id, next_state)

        elif current_state == "RE_WAITING_FOR_NAME":
            allowed_intents = ["NAME", "UNKNOWN"]
            
            if not customer_name:
                extracted = extract_customer_name_from_text(user_text, lang_code, agent_name=agent_name)
                if extracted:
                    customer_name = extracted
                    session_store.customer_name = extracted
                    meta["customer_name"] = customer_name
                    collected_info["customer_name"] = customer_name
                    await self.session_manager.update_session_metadata(
                        call_id, {"customer_name": customer_name, "collected_info": collected_info}
                    )

            if customer_name:
                detected_intent = "NAME"
                validated_intent = "NAME"
                target_template_name = "RE_PURPOSE_INTRO"
                next_state = "RE_INTEREST_DECISION"
            else:
                detected_intent = "UNKNOWN"
                validated_intent = "UNKNOWN"
                target_template_name = "RECOVERY"
                next_state = "RE_WAITING_FOR_NAME"

            await self.session_manager.update_session_state(call_id, next_state)

        elif current_state == "RE_INTEREST_DECISION":
            allowed_intents = ["CONFIRM_SITE_VISIT", "DECLINE_SITE_VISIT", "KNOWLEDGE_QUERY", "SELECT_BHK", "UNKNOWN"]
            t_lower = user_text.lower().strip().rstrip(".,!?।")
            
            decline_words = ["no", "nope", "not interested", "dont", "don't", "nahi", "na", "nahi chahiye", "वద్దు", "లేదు", "नहीं", "bye", "goodbye"]
            confirm_words = ["confirm", "yes", "yeah", "yep", "sure", "okay", "ok", "book", "visit", "schedule", "tomorrow", "weekend", "haan", "ha", "हाँ", "అవును", "సరే"]
            bhk_words = ["2 bhk", "3 bhk", "penthouse", "two bhk", "three bhk", "80 lakh", "1.2 crore", "2.5 crore", "bhk", "duplex", "flat", "apartment", "पेंटहाउस", "फ्लैट", "పెంట్‌హౌస్", "ఫ్లాట్"]
            
            is_decline = any(w in t_lower for w in decline_words)
            is_confirm = any(w in t_lower for w in confirm_words)
            is_bhk = any(w in t_lower for w in bhk_words)
            is_knowledge = is_real_estate_knowledge_query(user_text)
            
            pending_prompt = {
                "en": "\n\nWould you like to schedule a site visit to check out our model apartment?",
                "hi": "\n\nक्या आप हमारे मॉडल अपार्टमेंट को देखने के लिए साइट विजिट बुक करना चाहेंगे?",
                "te": "\n\nమా మోడల్ అపార్ట్‌మెంట్‌ను చూడటానికి సైట్ విజిట్ బుక్ చేయాలనుకుంటున్నారా?"
            }[lang_key]

            if is_decline:
                logger.info(f"[RE-INTENT] session={call_id} user declined site visit.")
                detected_intent = "DECLINE_SITE_VISIT"
                validated_intent = "DECLINE_SITE_VISIT"
                target_template_name = "RE_SITE_VISIT_DECLINE"
                next_state = "RE_CALL_ENDED"
                should_hangup = True
            elif is_confirm and not is_bhk:
                logger.info(f"[RE-INTENT] session={call_id} user confirmed site visit.")
                detected_intent = "CONFIRM_SITE_VISIT"
                validated_intent = "CONFIRM_SITE_VISIT"
                requested_tool = "book_site_visit"
                tool_allowed = validate_tool_call(industry, current_state, validated_intent, requested_tool, slots={"visit": user_text})
                if tool_allowed:
                    tool_executed = "book_site_visit"
                    tool_result = f"Success: Site visit booked for choice: {user_text}."
                target_template_name = "RE_SITE_VISIT_CONFIRM"
                next_state = "RE_CALL_ENDED"
                should_hangup = True
            elif is_bhk:
                choice = extract_bhk_choice(user_text, lang_key)
                logger.info(f"[RE-INTENT] session={call_id} user selected BHK choice='{choice}' (from raw: '{user_text}')")
                detected_intent = "SELECT_BHK"
                validated_intent = "SELECT_BHK"
                collected_info["unit_choice"] = choice
                await self.session_manager.update_session_metadata(call_id, {"collected_info": collected_info})
                target_template_name = "RE_RECOMMENDATION"
                next_state = "RE_SITE_VISIT_OFFER"
            elif is_knowledge:
                logger.info(f"[RE-KNOWLEDGE] session={call_id} query='{user_text}'")
                detected_intent = "KNOWLEDGE_QUERY"
                validated_intent = "KNOWLEDGE_QUERY"
                direct_ans = resolve_real_estate_direct_knowledge(user_text, lang_key)
                if direct_ans:
                    response_text = direct_ans + pending_prompt
                else:
                    facts = await self.rag_service.search_knowledge(campaign_id, user_text, limit=2)
                    if facts and facts[0].get("score", 0) >= 0.15:
                        response_text = facts[0]["text"] + pending_prompt
                    else:
                        logger.info(f"[RE-DYNAMIC-LLM] session={call_id} dynamic Gemini LLM generation for query='{user_text}'.")
                        llm_prompt = f"The customer asked: '{user_text}'. Answer their real estate question accurately, helpfully, and professionally in 1-2 sentences as the Skyline Developers sales executive."
                        llm_resp, _ = await self.llm_service.generate_completion(history + [{"role": "user", "content": llm_prompt}])
                        if llm_resp:
                            response_text = clean_speech_text(llm_resp) + pending_prompt
                        else:
                            response_text = {
                                "en": "Skyline Residency offers premium 2 & 3 BHK apartments and penthouses in Gachibowli with world-class amenities." + pending_prompt,
                                "hi": "Skyline Residency गच्चीबाउली में विश्व स्तरीय सुविधाओं के साथ 2 & 3 BHK और पेंटहाउस प्रदान करता है।" + pending_prompt,
                                "te": "Skyline Residency గచ్చిబౌలిలో ప్రపంచ స్థాయి సౌకర్యాలతో 2 & 3 BHK అపార్ట్‌మెంట్‌లను అందిస్తుంది।" + pending_prompt
                            }[lang_key]
                yield response_text, False, False
                bot_turn = {"role": "assistant", "content": response_text}
                await self.session_manager.append_message(call_id, bot_turn)
                await self.session_manager.update_session_state(call_id, current_state)
                return
            else:
                logger.info(f"[RE-INTENT] session={call_id} unclear input='{user_text}', asking for clarification.")
                detected_intent = "UNKNOWN"
                validated_intent = "UNCLEAR"
                response_text = {
                    "en": "Would you be interested in our 2 BHK, 3 BHK, or Penthouse options, or would you like to schedule a site visit?",
                    "hi": "क्या आप हमारे 2 BHK, 3 BHK या पेंटहाउस में रुचि रखते हैं, या एक साइट विजिट बुक करना चाहेंगे?",
                    "te": "మీరు 2 BHK, 3 BHK లేదా పెంట్‌హౌస్‌పై ఆసక్తి చూపుతున్నారా, లేదా సైట్ విజిట్ బుక్ చేయాలనుకుంటున్నారా?"
                }[lang_key]
                yield response_text, False, False
                bot_turn = {"role": "assistant", "content": response_text}
                await self.session_manager.append_message(call_id, bot_turn)
                await self.session_manager.update_session_state(call_id, current_state)
                return

            await self.session_manager.update_session_state(call_id, next_state)

        elif current_state == "RE_REQUIREMENT_COLLECTION":
            allowed_intents = ["PROVIDE_REQUIREMENT"]
            detected_intent = "PROVIDE_REQUIREMENT"
            validated_intent = "PROVIDE_REQUIREMENT"
            collected_info["requirements"] = user_text
            await self.session_manager.update_session_metadata(call_id, {"collected_info": collected_info})

            target_template_name = "RE_RECOMMENDATION"
            next_state = "RE_SITE_VISIT_OFFER"
            await self.session_manager.update_session_state(call_id, next_state)

        elif current_state == "RE_SITE_VISIT_OFFER":
            allowed_intents = ["CONFIRM_SITE_VISIT", "DECLINE_SITE_VISIT", "KNOWLEDGE_QUERY", "SELECT_BHK"]
            t_lower = user_text.lower().strip().rstrip(".,!?।")
            requested_tool = "book_site_visit"
            confirm_words = ["confirm", "yes", "yeah", "yep", "sure", "okay", "ok", "haan", "ha", "हाँ", "అవును", "visit", "book", "tomorrow", "weekend", "schedule"]
            decline_words = ["no", "nope", "not interested", "dont", "don't", "nahi", "na", "nahi chahiye", "వద్దు", "లేదు", "नहीं", "bye"]
            bhk_words = ["2 bhk", "3 bhk", "penthouse", "two bhk", "three bhk", "duplex", "bhk", "पेंटहाउस", "फ्लैट", "పెంట్‌హౌస్", "ఫ్లాట్"]
            
            is_decline = any(w in t_lower for w in decline_words)
            is_confirm = any(w in t_lower for w in confirm_words)
            is_bhk = any(w in t_lower for w in bhk_words)
            is_knowledge = is_real_estate_knowledge_query(user_text)

            if is_decline:
                detected_intent = "DECLINE_SITE_VISIT"
                validated_intent = "DECLINE_SITE_VISIT"
                target_template_name = "RE_SITE_VISIT_DECLINE"
                next_state = "RE_CALL_ENDED"
                should_hangup = True
            elif is_confirm and not is_bhk:
                detected_intent = "CONFIRM_SITE_VISIT"
                validated_intent = "CONFIRM_SITE_VISIT"
                tool_allowed = validate_tool_call(industry, current_state, validated_intent, requested_tool, slots={"visit": user_text})
                if tool_allowed:
                    tool_executed = "book_site_visit"
                    tool_result = f"Success: Site visit booked."
                target_template_name = "RE_SITE_VISIT_CONFIRM"
                next_state = "RE_CALL_ENDED"
                should_hangup = True
            elif is_bhk:
                choice = extract_bhk_choice(user_text, lang_key)
                detected_intent = "SELECT_BHK"
                validated_intent = "SELECT_BHK"
                collected_info["unit_choice"] = choice
                await self.session_manager.update_session_metadata(call_id, {"collected_info": collected_info})
                target_template_name = "RE_RECOMMENDATION"
                next_state = "RE_SITE_VISIT_OFFER"
            elif is_knowledge:
                direct_ans = resolve_real_estate_direct_knowledge(user_text, lang_key)
                pending_prompt = {
                    "en": "\n\nWould you like to schedule a site visit to check out the property?",
                    "hi": "\n\nक्या आप प्रॉपर्टी देखने के लिए साइट विजिट बुक करना चाहेंगे?",
                    "te": "\n\nప్రాపర్టీని చూడటానికి సైట్ విజిట్ బుక్ చేయాలనుకుంటున్నారా?"
                }[lang_key]
                if direct_ans:
                    response_text = direct_ans + pending_prompt
                else:
                    facts = await self.rag_service.search_knowledge(campaign_id, user_text, limit=2)
                    if facts and facts[0].get("score", 0) >= 0.15:
                        response_text = facts[0]["text"] + pending_prompt
                    else:
                        logger.info(f"[RE-DYNAMIC-LLM] session={call_id} dynamic Gemini LLM generation for query='{user_text}'.")
                        llm_prompt = f"The customer asked: '{user_text}'. Answer their real estate question accurately, helpfully, and professionally in 1-2 sentences as the Skyline Developers sales executive."
                        llm_resp, _ = await self.llm_service.generate_completion(history + [{"role": "user", "content": llm_prompt}])
                        if llm_resp:
                            response_text = clean_speech_text(llm_resp) + pending_prompt
                        else:
                            response_text = {
                                "en": "Skyline Residency offers luxury 2 & 3 BHK apartments in Gachibowli with ready sample flats." + pending_prompt,
                                "hi": "Skyline Residency गच्चीबाउली में रेडी सैंपल फ्लैट्स के साथ 2 & 3 BHK प्रदान करता है।" + pending_prompt,
                                "te": "Skyline Residency గచ్చిబౌలిలో రెడీ శాంపిల్ ఫ్లాట్లతో 2 & 3 BHK ఆఫర్ చేస్తుంది." + pending_prompt
                            }[lang_key]
                yield response_text, False, False
                bot_turn = {"role": "assistant", "content": response_text}
                await self.session_manager.append_message(call_id, bot_turn)
                await self.session_manager.update_session_state(call_id, current_state)
                return
            else:
                target_template_name = "RE_SITE_VISIT_CONFIRM"
                next_state = "RE_CALL_ENDED"
                should_hangup = True

            await self.session_manager.update_session_state(call_id, next_state)

    meta["turn_detected_intent"] = detected_intent
    meta["turn_validated_intent"] = validated_intent
    meta["turn_response_policy"] = "template" if target_template_name else "llm_fallback"
    meta["turn_tool_executed"] = tool_executed or "None"
    meta["turn_tool_allowed"] = tool_allowed
    await self.session_manager.update_session_metadata(call_id, meta)

    try:
        logger.info(
            f"[CONVERSATION-GUARD]\n"
            f"session_id={call_id}\n"
            f"industry={industry}\n"
            f"current_state={current_state}\n"
            f"customer_name={customer_name or 'UNKNOWN'}\n"
            f"last_agent_message='{last_agent_message}'\n"
            f"customer_transcript='{user_text}'\n"
            f"detected_intent={detected_intent}\n"
            f"allowed_intents={allowed_intents}\n"
            f"validated_intent={validated_intent}\n"
            f"next_state={next_state}\n"
            f"tool_allowed={str(tool_allowed).lower()}\n"
            f"requested_tool={requested_tool or 'None'}"
        )
    except Exception as ge:
        logger.warning(f"Conversation guard logging warning: {ge}")

    if target_template_name:
        if target_template_name == "REPEAT":
            target_text = last_agent_message or "Could you please confirm, reschedule, or cancel your appointment?"
        else:
            template_text = TEMPLATES[industry][target_template_name][lang_key]
            target_text = template_text.format(
                agent_name=agent_name,
                customer_name=customer_name or "",
                slot=collected_info.get("reschedule_slot", "tomorrow"),
                unit_choice=collected_info.get("unit_choice", "premium 2 BHK apartment" if lang_key == "en" else "प्रीमियम 2 BHK फ्लैट")
            )
        target_text = target_text.replace("Hi ,", "Hi,").replace("Hello ,", "Hello,").replace("  ", " ").strip()

        # --- ZERO-LLM FAST PATH ---
        # The template text is already fully resolved above. Sending it through the LLM
        # just to echo it back as JSON costs ~1000ms TTFT every turn. Skip the LLM entirely
        # and stream the text word-by-word directly to TTS for instant response.
        logger.info(f"[TEMPLATE-FAST-PATH] Bypassing LLM for template={target_template_name} text_chars={len(target_text)}")

        words = target_text.split(" ")
        full_text = target_text

        # Sanity check — no metadata bleed
        if "[" in full_text or "]" in full_text or "customer_name=" in full_text:
            logger.error(f"[TTS-SANITIZER-REJECT] Metadata in template text: '{full_text}'")
            full_text = get_deterministic_fallback(industry, current_state, lang_key, customer_name, collected_info)
            yield full_text, False, False
        else:
            is_valid, fallback_val = validate_response_against_state(full_text, next_state, industry, customer_name)
            if not is_valid:
                logger.warning(f"[RESPONSE-VALIDATION] State contract violation. Replacing with fallback '{fallback_val}'")
                full_text = fallback_val
                yield fallback_val, False, False
            else:
                # Pre-split at sentence boundaries so TTS can dispatch each sentence
                # to synthesis immediately rather than accumulating all words first.
                # e.g. "Hello Krish." (12 chars) becomes sentence-1 instantly (~250ms synth)
                # while the longer follow-up synthesizes in parallel.
                # Split on sentence terminators while keeping the terminator attached
                raw_sentences = re.split(r'(?<=[.!?।])\s+', target_text)
                for i, sent in enumerate(raw_sentences):
                    sent = sent.strip()
                    if not sent:
                        continue
                    # Yield sentence with a trailing space so TTS splitter sees a word boundary
                    chunk = sent if i == 0 else " " + sent
                    yield chunk, False, False
                    # Tiny yield point lets the TTS producer task run and dispatch synthesis
                    # for this sentence before we push the next one
                    await asyncio.sleep(0)

        if full_text:
            bot_turn = {"role": "assistant", "content": full_text}
            await self.session_manager.append_message(call_id, bot_turn)

    else:
        compiled_prompt, _ = await self.prompt_service.build_prompt(
            campaign_id,
            industry=industry,
            language=lang_str,
            agent_name=agent_name,
            current_state=current_state,
            collected_info=collected_info,
            user_text=user_text
        )

        history_dialogue = [m for m in history if m["role"] in ("user", "assistant")]
        if user_text != "[CALL_START]":
            user_turn = {"role": "user", "content": user_text}
            history_dialogue.append(user_turn)
            await self.session_manager.append_message(call_id, user_turn)

        messages_to_send = [{"role": "system", "content": compiled_prompt}] + history_dialogue
        llm_stream = self.llm_service.generate_completion_stream(messages_to_send, None)
        speech_stream = extract_speech_from_json_stream(llm_stream)

        full_text_acc = []
        async for chunk in speech_stream:
            if chunk:
                if "[" in chunk or "]" in chunk or "{" in chunk or "}" in chunk or "customer_name=" in chunk:
                    logger.error(f"[TTS-SANITIZER-REJECT] Metadata leak in chunk: '{chunk}'")
                    full_text_acc = [get_deterministic_fallback(industry, current_state, lang_key, customer_name, collected_info)]
                    break
                full_text_acc.append(chunk)
                yield chunk, False, False

        full_text = "".join(full_text_acc).strip()
        
        is_valid, fallback_val = validate_response_against_state(full_text, next_state, industry, customer_name)
        if not is_valid:
            logger.warning(f"[RESPONSE-VALIDATION] State contract violation. Replacing '{full_text}' with fallback '{fallback_val}'")
            full_text = fallback_val
            yield fallback_val, False, False

        if full_text:
            bot_turn = {"role": "assistant", "content": full_text}
            await self.session_manager.append_message(call_id, bot_turn)

    logger.info(
        f"[TURN] session={call_id} state={next_state} agent={agent_name} "
        f"language={lang_code} customer_name={customer_name or 'UNKNOWN'} "
        f"tool={tool_executed or 'None'} tool_result={tool_result or 'None'} "
        f"should_hangup={should_hangup}"
    )

    yield None, should_hangup, should_transfer


ConversationEngine.process_voice_demo_turn_stream = _process_voice_demo_turn_stream_impl

