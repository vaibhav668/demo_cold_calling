import uuid
import re
from typing import Optional, Dict, Any, Tuple
from app.services.rag_service import RAGService

# Define static goals for database-less demo fallback mode
HOSPITAL_STATE_GOALS = {
    "GREETING": (
        "Greet the customer naturally and ask for their name immediately.\n"
        "English: 'Hi, this is Maya from CityCare Hospital. May I know whom I'm speaking with?'\n"
        "Instructions: Do NOT speak about appointments, doctors, dates, or timings yet. Ask ONLY for their name.\n"
        "Transition tag: [STATE: WAIT_FOR_NAME]"
    ),
    "GREETING_HINDI": (
        "नमस्ते! मैं सिटीकेयर हॉस्पिटल से माया बात कर रही हूँ। क्या मैं आपका नाम जान सकती हूँ?\n"
        "Instructions: अपॉइंटमेंट के बारे में अभी बात न करें। केवल नाम पूछें।\n"
        "Transition tag: [STATE: WAIT_FOR_NAME]"
    ),
    "GREETING_TELUGU": (
        "నమస్కారం! నేను సిటీకేర్ హాస్పిటల్ నుండి మాయ మాట్లాడుతున్నాను. మీ పేరు తెలుసుకోవచ్చా?\n"
        "Instructions: అపాయింట్‌మెంట్ గురించి ఇప్పుడే మాట్లాడకండి. కేవలం పేరు అడగండి.\n"
        "Transition tag: [STATE: WAIT_FOR_NAME]"
    ),

    "WAIT_FOR_NAME": (
        "If the customer provided their name in the input:\n"
        "  - Acknowledge their name, thank them, and state the purpose of the call: appointment scheduled tomorrow with Dr. Sharma.\n"
        "  - Ask if they want to confirm, reschedule, or cancel.\n"
        "  - Transition tag: [STATE: WAIT_FOR_DECISION] [EXTRACT: customer_name=<extracted_name>]\n"
        "If the customer did NOT provide their name or it is unclear:\n"
        "  - Politely ask for their name once more.\n"
        "  - Transition tag: [STATE: WAIT_FOR_NAME]"
    ),

    "PURPOSE_OF_CALL": (
        "Acknowledge customer by name and state call purpose: appointment scheduled tomorrow with Dr. Sharma. Ask if they want to confirm, reschedule, or cancel.\n"
        "English: 'Great, thank you {{customer_name}}. I'm calling regarding your appointment with Dr. Sharma tomorrow. Would you like to confirm, reschedule, or cancel?'\n"
        "Transition tag: [STATE: WAIT_FOR_DECISION]"
    ),
    "PURPOSE_OF_CALL_HINDI": (
        "नाम का अभिवादन करें और उद्देश्य बताएं: कल डॉ. शर्मा के साथ अपॉइंटमेंट। पूछें कि कन्फर्म, रीशेड्यूल या कैंसिल करना है।\n"
        "Hindi: 'धन्यवाद {{customer_name}}। मैं कल डॉ. शर्मा के साथ आपके अपॉइंटमेंट के सिलसिले में कॉल कर रही हूँ। क्या आप इसे कन्फर्म करना चाहेंगे, रीशेड्यूल करना चाहेंगे या कैंसिल?'\n"
        "Transition tag: [STATE: WAIT_FOR_DECISION]"
    ),
    "PURPOSE_OF_CALL_TELUGU": (
        "పేరుతో ధన్యవాదాలు చెప్పి కాల్ ఉద్దేశ్యం చెప్పండి: రేపు డాక్టర్ శర్మతో అపాయింట్‌మెంట్. కన్ఫర్మ్, రీషెడ్యూల్ లేదా క్యాన్సిల్ చేయాలా అని అడగండి.\n"
        "Telugu: 'ధన్యవాదాలు {{customer_name}}. రేపు డాక్టర్ శర్మ గారితో ఉన్న మీ అపాయింట్‌మెంట్ గురించి కాల్ చేస్తున్నాను. మీరు దీన్ని కన్ఫర్మ్ చేయాలనుకుంటున్నారా, రీషెడ్యూల్ చేయాలనుకుంటున్నారా లేదా క్యాన్సిల్ చేయాలనుకుంటున్నారా?'\n"
        "Transition tag: [STATE: WAIT_FOR_DECISION]"
    ),

    "PROCESS_CONFIRM": (
        "Confirm the appointment and ask if they need directions or have any questions.\n"
        "English: 'Perfect, I have confirmed your appointment for tomorrow at 11 AM. Do you need the hospital address or directions?'\n"
        "Transition tag: [STATE: CLOSING]"
    ),
    "PROCESS_CONFIRM_HINDI": (
        "अपॉइंटमेंट की पुष्टि करें।\n"
        "Hindi: 'बहुत बढ़िया, मैंने कल सुबह 11 बजे के लिए आपका अपॉइंटमेंट कन्फर्म कर दिया है। क्या आपको हॉस्पिटल का पता चाहिए?'\n"
        "Transition tag: [STATE: CLOSING]"
    ),
    "PROCESS_CONFIRM_TELUGU": (
        "అపాయింట్‌మెంట్ కన్ఫర్మ్ చేయండి.\n"
        "Telugu: 'చాలా మంచిది, రేపు ఉదయం 11 గంటలకు మీ అపాయింట్‌మెంట్ కన్ఫర్మ్ చేయబడింది. మీకు హాస్పిటల్ అడ్రస్ వివరాలు కావాలా?'\n"
        "Transition tag: [STATE: CLOSING]"
    ),

    "PROCESS_CANCEL": (
        "Confirm the cancellation, express regret, and say the slot is released.\n"
        "English: 'I've cancelled your appointment. We've released the slot. Hope to serve you next time.'\n"
        "Transition tag: [STATE: CLOSING]"
    ),
    "PROCESS_CANCEL_HINDI": (
        "रद्दीकरण की पुष्टि करें।\n"
        "Hindi: 'मैंने आपका अपॉइंटमेंट कैंसिल कर दिया है। आशा है अगली बार आपकी सेवा का मौका मिलेगा।'\n"
        "Transition tag: [STATE: CLOSING]"
    ),
    "PROCESS_CANCEL_TELUGU": (
        "క్యాన్సిలేషన్ కన్ఫర్మ్ చేయండి.\n"
        "Telugu: 'నేను మీ అపాయింట్‌మెంట్‌ని క్యాన్సిల్ చేసాను. మళ్లీ ఎప్పుడైనా మీకు సేవ చేసే అవకాశం లభిస్తుందని ఆశిస్తున్నాము.'\n"
        "Transition tag: [STATE: CLOSING]"
    ),

    "PROCESS_RESCHEDULE": (
        "Ask the customer for their preferred date or time slot to reschedule.\n"
        "English: 'No problem, what date or time slot works best for you tomorrow or next week?'\n"
        "Transition tag: [STATE: CAPTURE_RESCHEDULE_SLOT]"
    ),
    "PROCESS_RESCHEDULE_HINDI": (
        "रीशेड्यूल के लिए पसंदीदा समय पूछें।\n"
        "Hindi: 'कोई बात नहीं, कल या अगले हफ्ते के लिए आपके लिए कौन सा समय सही रहेगा?'\n"
        "Transition tag: [STATE: CAPTURE_RESCHEDULE_SLOT]"
    ),
    "PROCESS_RESCHEDULE_TELUGU": (
        "రీషెడ్యూల్ కోసం సమయం అడగండి.\n"
        "Telugu: 'పర్వాలేదండి, రేపు లేదా వచ్చే వారంలో ఏ రోజు మరియు ఏ సమయం మీకు వీలుగా ఉంటుంది?'\n"
        "Transition tag: [STATE: CAPTURE_RESCHEDULE_SLOT]"
    ),

    "CONFIRM_RESCHEDULE_SLOT": (
        "Confirm the new rescheduled slot chosen by customer.\n"
        "English: 'Got it, I have successfully rescheduled your appointment to {{reschedule_slot}}. Do you need anything else?'\n"
        "Transition tag: [STATE: CLOSING]"
    ),
    "CONFIRM_RESCHEDULE_SLOT_HINDI": (
        "Hindi: 'ठीक है, मैंने आपका अपॉइंटमेंट {{reschedule_slot}} पर रीशेड्यूल कर दिया है। क्या आपको कुछ और मदद चाहिए?'\n"
        "Transition tag: [STATE: CLOSING]"
    ),
    "CONFIRM_RESCHEDULE_SLOT_TELUGU": (
        "Telugu: 'సరేనండి, నేను మీ అపాయింట్‌మెంట్‌ను {{reschedule_slot}} కు విజయవంతంగా రీషెడ్యూల్ చేసాను. మీకు ఇంకా ఏదైనా సహాయం కావాలా?'\n"
        "Transition tag: [STATE: CLOSING]"
    ),

    "CLOSING": (
        "Deliver a warm, professional goodbye. Do NOT ask any more questions.\n"
        "Outcome examples:\n"
        "  Confirmed: 'Thank you {{customer_name}}, we look forward to seeing you tomorrow. Take care, goodbye!'\n"
        "  Rescheduled: 'Thank you {{customer_name}}, your new slot is confirmed. Have a wonderful day, goodbye!'\n"
        "  Cancelled: 'Take care, goodbye!'\n"
        "This is the FINAL turn. Do NOT wait for another reply.\n"
        "Transition tag: [STATE: END_CALL]"
    ),
    "CLOSING_HINDI": (
        "अलविदा कहें। कोई और प्रश्न न पूछें।\n"
        "Hindi: 'धन्यवाद {{customer_name}}, कल आपसे मिलकर ख़ुशी होगी। अपना ख्याल रखें, अलविदा!'\n"
        "Transition tag: [STATE: END_CALL]"
    ),
    "CLOSING_TELUGU": (
        "సెలవు తీసుకోండి. ఇక ప్రశ్నలు అడగకండి.\n"
        "Telugu: 'ధన్యవాదాలు {{customer_name}}, రేపు మిమ్మల్ని కలుసుకోవడానికి ఎదురుచూస్తున్నాము. జాగ్రత్త, సెలవు!'\n"
        "Transition tag: [STATE: END_CALL]"
    ),
    "END_CALL": (
        "The call has concluded. Do not speak anything.\n"
        "Transition tag: [STATE: END_CALL]"
    )
}

REAL_ESTATE_STATE_GOALS = {
    "GREETING": (
        "Greet the customer naturally and ask for their name immediately.\n"
        "English: 'Hi, this is Ananya from Skyline Developers. May I know whom I'm speaking with?'\n"
        "Instructions: Do NOT speak about properties, locations, or budgets yet. Ask ONLY for their name.\n"
        "Transition tag: [STATE: WAIT_FOR_NAME]"
    ),
    "GREETING_HINDI": (
        "नमस्ते! मैं स्काईलाइन डेवलपर्स से अनन्या बात कर रही हूँ। क्या मैं आपका नाम जान सकती हूँ?\n"
        "Instructions: अभी प्रॉपर्टी के बारे में बात न करें। केवल नाम पूछें।\n"
        "Transition tag: [STATE: WAIT_FOR_NAME]"
    ),
    "GREETING_TELUGU": (
        "నమస్కారం! నేను స్కైలైన్ డెవలపర్స్ నుండి అనన్య మాట్లాడుతున్నాను. మీ పేరు తెలుసుకోవచ్చా?\n"
        "Instructions: ఇప్పుడే ప్రాపర్టీల గురించి మాట్లాడకండి. కేవలం పేరు అడగండి.\n"
        "Transition tag: [STATE: WAIT_FOR_NAME]"
    ),

    "WAIT_FOR_NAME": (
        "If the customer provided their name in the input:\n"
        "  - Acknowledge their name, thank them, and introduce the new premium project in Gachibowli starting at 80 Lakhs.\n"
        "  - Ask if they are looking to buy or invest in a property recently.\n"
        "  - Transition tag: [STATE: INTEREST_CHECK] [EXTRACT: customer_name=<extracted_name>]\n"
        "If the customer did NOT provide their name or it is unclear:\n"
        "  - Politely ask for their name once more.\n"
        "  - Transition tag: [STATE: WAIT_FOR_NAME]"
    ),

    "PURPOSE_OF_CALL": (
        "State call purpose: Pitch Skyline Residency premium properties starting at 80L in Gachibowli, Hyderabad. Ask if they are looking for a property right now.\n"
        "English: 'Great, thank you {{customer_name}}. I'm calling to introduce our new premium project in Gachibowli, featuring 2 and 3 BHK luxury apartments starting at 80 Lakhs. Are you looking to buy or invest in a property recently?'\n"
        "Transition tag: [STATE: INTEREST_CHECK]"
    ),
    "PURPOSE_OF_CALL_HINDI": (
        "Hindi: 'धन्यवाद {{customer_name}}। मैं गचीबोवली में हमारे नए लग्जरी प्रोजेक्ट के बारे में जानकारी देने के लिए कॉल कर रही हूँ, जहाँ 2 और 3 BHK फ्लैट्स 80 लाख से शुरू हैं। क्या आप अभी नया घर खरीदने का मन बना रहे हैं?'\n"
        "Transition tag: [STATE: INTEREST_CHECK]"
    ),
    "PURPOSE_OF_CALL_TELUGU": (
        "Telugu: 'ధన్యవాదాలు {{customer_name}}. గచ్చిబౌలిలోని మా కొత్త ప్రీమియం ప్రాజెక్ట్ గురించి మీకు తెలియజేయడానికి కాల్ చేసాను, ఇక్కడ 2 & 3 BHK లగ్జరీ అపార్ట్‌మెంట్‌లు 80 లక్షల నుండి ప్రారంభమవుతాయి. మీరు ప్రస్తుతం ఇల్లు కొనే ఆలోచనలో ఉన్నారా?'\n"
        "Transition tag: [STATE: INTEREST_CHECK]"
    ),

    "INTEREST_CHECK": (
        "If they are interested, proceed to ask if they prefer 2 BHK or 3 BHK. If NOT interested, route directly to closing.\n"
        "English: 'Wonderful! Do you prefer a 2 BHK or a larger 3 BHK configuration?'\n"
        "Transition tag: [STATE: QUALIFICATION]"
    ),
    "INTEREST_CHECK_HINDI": (
        "Hindi: 'बहुत बढ़िया! आपको 2 BHK फ्लैट पसंद आएगा या बड़ा 3 BHK फ्लैट?'\n"
        "Transition tag: [STATE: QUALIFICATION]"
    ),
    "INTEREST_CHECK_TELUGU": (
        "Telugu: 'చాలా సంతోషం! మీరు 2 BHK లేదా 3 BHK అపార్ట్‌మెంట్‌ని ఇష్టపడుతున్నారా?'\n"
        "Transition tag: [STATE: QUALIFICATION]"
    ),

    "QUALIFICATION": (
        "Pitch the site visit or consulting call: 'Would you be interested in scheduling a site visit or speaking with one of our property consultants?'\n"
        "Transition tag: [STATE: CLOSING]"
    ),
    "CLOSING": (
        "Deliver a warm, professional goodbye. Do NOT ask any more questions.\n"
        "Address the customer by name one final time if you know it.\n"
        "Based on their outcome: summarize their interest or decision in one sentence, then thank them.\n"
        "Outcome examples:\n"
        "  Interested: 'Thank you {{customer_name}}. I've noted your interest in a 3-bedroom apartment. One of our consultants will be in touch shortly. Have a wonderful day. Goodbye!'\n"
        "  Not interested: 'Absolutely, no problem at all. Thank you for your time. Do reach out if you ever need assistance. Have a great day. Goodbye!'\n"
        "This is the FINAL turn. Do NOT wait for another reply.\n"
        "Transition tag: [STATE: END_CALL]"
    ),
    "END_CALL": (
        "The call has concluded. Do not speak anything. The session will now be terminated automatically.\n"
        "Transition tag: [STATE: END_CALL]"
    )
}

BASE_TEMPLATE = (
    "You are {{agent_name}}, a professional, confident, and warm representative of {{company_name}}.\n"
    "Your absolute goal is to behave like a trained, human outbound agent making a genuine cold call. "
    "Do NOT sound like a voice assistant, a chatbot, or a robotic agent. Speak naturally, brief (ideal length 5-18 words, occasionally 25 words), "
    "react intelligently to interruptions, and address the customer by name occasionally.\n"
    "\n"
    "### CURRENT CONVERSATION STATE\n"
    "Current State: {{current_state}}\n"
    "State Goal: {{state_goal}}\n"
    "\n"
    "### COLLECTED INFORMATION SO FAR\n"
    "{{collected_info_text}}\n"
    "\n"
    "### BUSINESS RULES & FLOW\n"
    "{{business_rules}}\n"
    "\n"
    "### END-OF-TURN OUTPUT TAGGING RULE (MANDATORY)\n"
    "At the very end of your response, and ONLY at the end, append the next logical state and any newly extracted information from the customer's input.\n"
    "Format: `[STATE: <next_state>] [EXTRACT: key1=value1, key2=value2]`\n"
    "Example: If customer said they are Rahul and you verify them, write: \n"
    "'Thank you Rahul. I'm calling from CityCare Hospital... [STATE: QUALIFICATION] [EXTRACT: customer_name=Rahul]'\n"
    "Only update keys when new details are provided. Do NOT output markdown, formatting symbols, or JSON."
)

LANGUAGE_TEMPLATES = {
    "English": (
        "Guidelines for English Speech:\n"
        "- Maintain natural human sentence pacing. Use pauses (indicated with commas and ellipses) for a relaxed flow.\n"
        "- Use contractions naturally (e.g. 'I'll' instead of 'I will', 'Who's this' instead of 'Who is this')."
    ),
    "Hindi": (
        "Guidelines for Hindi Speech:\n"
        "- Write strictly in Hindi language using Devanagari script. Do NOT use English alphabets or Roman text.\n"
        "- Speak naturally as a native speaker would. Do not translate word-by-word from English.\n"
        "- Use native greetings and natural filler/acknowledgement words: 'नमस्ते', 'बिल्कुल', 'मैं समझ सकता हूँ', 'यह बहुत अच्छा सवाल है'."
    ),
    "Telugu": (
        "Guidelines for Telugu Speech:\n"
        "- Write strictly in Telugu language using Telugu script. Do NOT use English alphabets or Roman text.\n"
        "- Speak naturally as a native speaker would. Do not translate word-by-word from English.\n"
        "- Use native greetings and natural filler/acknowledgement words: 'నమస్కారం', 'తప్పకుండా', 'నేను అర్థం చేసుకోగలను', 'ఇది చాలా మంచి ప్రశ్న'."
    )
}

class PromptService:
    def __init__(self, db: Optional[Any] = None):
        self.db = db
        self.rag_service = RAGService()
        self.template_repo = None
        self.customer_repo = None

        if db is not None:
            try:
                from app.repositories.prompt_template import PromptTemplateRepository
                from app.repositories.customer import CustomerRepository
                self.template_repo = PromptTemplateRepository(db)
                self.customer_repo = CustomerRepository(db)
            except ImportError:
                pass

    def _replace_placeholders(self, text: Optional[str], variables: Dict[str, Any]) -> str:
        """Replace all curly brace placeholders {{var}} with resolved values."""
        if not text:
            return ""
        def replacement(match):
            key = match.group(1).strip()
            return str(variables.get(key, ""))
        return re.sub(r"\{\{([^}]+)\}\}", replacement, text)

    async def build_prompt(
        self,
        campaign_id: uuid.UUID,
        *args,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Compile dynamic conversation prompt resolving template placeholders and appending RAG facts."""
        customer_id = kwargs.get("customer_id")
        rag_query = kwargs.get("rag_query")
        session_id = kwargs.get("session_id")
        industry = kwargs.get("industry")
        language = kwargs.get("language")
        agent_name = kwargs.get("agent_name")
        current_state = kwargs.get("current_state", "GREETING")
        collected_info = kwargs.get("collected_info")

        if args:
            if isinstance(args[0], uuid.UUID) or (isinstance(args[0], str) and len(args[0]) == 36 and "-" in args[0]):
                # Signature 1: customer_id, rag_query=None, session_id=None
                customer_id = args[0]
                if len(args) > 1:
                    rag_query = args[1]
                if len(args) > 2:
                    session_id = args[2]
            else:
                # Signature 2: industry, language, agent_name, current_state="GREETING", collected_info=None, rag_query=None
                industry = args[0]
                language = args[1]
                agent_name = args[2]
                if len(args) > 3:
                    current_state = args[3]
                if len(args) > 4:
                    collected_info = args[4]
                if len(args) > 5:
                    rag_query = args[5]

        # Mode A: Database-backed mode (ai_cold_call)
        if self.db is not None and self.template_repo is not None and customer_id is not None:
            try:
                from app.core.exceptions import NotFoundException
            except ImportError:
                class NotFoundException(Exception):
                    pass

            template = await self.template_repo.get_active_by_campaign(campaign_id)
            if not template:
                raise NotFoundException("Active prompt template not configured for this campaign.")
                
            customer = await self.customer_repo.get(customer_id)
            if not customer:
                raise NotFoundException("Customer not found.")
                
            variables = {
                "first_name": customer.first_name or "",
                "last_name": customer.last_name or "",
                "email": customer.email or "",
                "phone_number": customer.phone_number or "",
                "id": str(customer.id)
            }
            
            if customer.custom_variables and isinstance(customer.custom_variables, dict):
                for k, v in customer.custom_variables.items():
                    variables[k] = v
                    
            # Resolve dynamic identity variables
            agent_name = "Sophia"
            hospital_name = "CityCare Hospital"
            builder = "Skyline Developers"
            property_name = "3 BHK Apartment"

            if session_id:
                from app.services.session_manager import SessionManager
                sm = SessionManager()
                session_meta = await sm.get_session_metadata(session_id)
                if session_meta:
                    agent_name = session_meta.get("agent_name", "Sophia")

            variables["agent_name"] = agent_name
            variables["hospital_name"] = hospital_name
            variables["builder"] = builder
            variables["property_name"] = property_name

            compiled_sys = self._replace_placeholders(template.system_prompt, variables)
            compiled_goals = self._replace_placeholders(template.conversation_goals, variables)
            compiled_lang = self._replace_placeholders(template.language_prompt, variables)
            
            # Replace hardcoded values to match runtime identity selections
            compiled_sys = compiled_sys.replace("Sarah", agent_name).replace("James", agent_name)
            compiled_sys = compiled_sys.replace("Mercy Hospital", hospital_name)
            compiled_sys = compiled_sys.replace("Premium Realty", builder)

            compiled_goals = compiled_goals.replace("Sarah", agent_name).replace("James", agent_name)
            compiled_goals = compiled_goals.replace("Mercy Hospital", hospital_name)
            compiled_goals = compiled_goals.replace("Premium Realty", builder)

            prompt_parts = [
                "### SYSTEM ROLE & INSTRUCTIONS",
                compiled_sys
            ]
            
            if compiled_goals:
                prompt_parts.extend([
                    "",
                    "### CONVERSATION GOALS",
                    compiled_goals
                ])
                
            if compiled_lang:
                prompt_parts.extend([
                    "",
                    "### STYLE & LANGUAGE GUIDELINES",
                    compiled_lang
                ])
                
            # Skip local SentenceTransformer model load on the CALL_START greeting initialization to prevent websocket timeouts
            if rag_query and rag_query != "[CALL_START]":
                facts = await self.rag_service.search_knowledge(campaign_id, rag_query, limit=3)
                if facts:
                    facts_text = "\n".join([f"- {item['text']}" for item in facts])
                    prompt_parts.extend([
                        "",
                        "### RETRIEVED KNOWLEDGE BASE FACTS",
                        facts_text
                    ])
                    
            final_prompt = "\n".join(prompt_parts)
            return final_prompt, variables

        # Mode B: Database-less fallback mode (demo_cold_calling)
        else:
            collected_info = collected_info or {}
            
            variables = {
                "agent_name": agent_name or "Ananya",
                "preferred_language": language or "English",
                "current_state": current_state
            }

            # Resolve state goals and company info
            if (industry or "").lower() == "hospital":
                variables["company_name"] = "CityCare Hospital"
                state_goal_template = HOSPITAL_STATE_GOALS.get(current_state, HOSPITAL_STATE_GOALS["GREETING"])
            else:
                variables["company_name"] = "Skyline Developers"
                state_goal_template = REAL_ESTATE_STATE_GOALS.get(current_state, REAL_ESTATE_STATE_GOALS["GREETING"])

            variables["state_goal"] = self._replace_placeholders(state_goal_template, variables)

            # Build collected info summary text
            info_lines = []
            for k, v in collected_info.items():
                info_lines.append(f"- {k}: {v}")
            variables["collected_info_text"] = "\n".join(info_lines) if info_lines else "- No details collected yet."

            # Compile RAG facts and business rules
            business_rules_list = []
            if rag_query and rag_query != "[CALL_START]":
                facts = await self.rag_service.search_knowledge(campaign_id, rag_query, limit=3)
                if facts:
                    for idx, fact in enumerate(facts):
                        business_rules_list.append(f"Fact {idx+1}: {fact['text']}")
            
            if not business_rules_list:
                if (industry or "").lower() == "hospital":
                    business_rules_list.append("Rule: Outbound Appointment confirmation call regarding Dr. Sharma scheduled tomorrow at 11 AM.")
                else:
                    business_rules_list.append("Rule: Outbound sales call regarding Orchard Heights premium 2/3/4 BHK starting at 80L in Gachibowli.")

            # Out-of-RAG/fallback instructions
            if (industry or "").lower() == "hospital":
                business_rules_list.append("Fallback: If customer asks questions unavailable in facts, say exactly: 'I don't have the exact information available at the moment, but our hospital staff would be happy to assist you further.'")
            else:
                business_rules_list.append("Fallback: If customer asks questions unavailable in facts, say exactly: 'I don't have the exact information available right now, but our sales specialist can certainly help with that.'")

            variables["business_rules"] = "\n".join(business_rules_list)

            # Assemble prompt parts
            compiled_base = self._replace_placeholders(BASE_TEMPLATE, variables)
            lang_guidelines = LANGUAGE_TEMPLATES.get(language or "English", LANGUAGE_TEMPLATES["English"])

            prompt_parts = [
                compiled_base,
                "",
                "### STYLE & NATIVE SPEECH GUIDELINES",
                lang_guidelines,
                "",
                "### CRITICAL TTS FORMATTING CONSTRAINTS",
                "- Always write in short, easily digestible sentences. Ideal response length: 5 to 18 words, occasionally 25 words.",
                "- Use ellipses (...) or commas (,) to encourage natural voice pauses in EdgeTTS.",
                "- NEVER use markdown formatting like asterisks (bold) or hashes (headers) in response speech.",
                "- Output only the direct dialogue response that the voice agent will speak followed by the [STATE: ...] and [EXTRACT: ...] tags.",
                "- Keep tags separated from the speech text so they can be parsed out."
            ]

            final_prompt = "\n".join(prompt_parts)
            return final_prompt, variables
