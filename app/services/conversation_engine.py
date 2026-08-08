import uuid
import json
import re
from typing import Tuple, List, Dict, Any, Optional, AsyncGenerator
from app.services.session_manager import SessionManager
from app.services.llm_service import LLMService
from app.services.prompt_service import PromptService
from app.services.rag_service import RAGService
from app.core.logging import logger

HAS_DB = True
AsyncSession = None
CallLogRepository = None
CallLog = None

try:
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.repositories.call_log import CallLogRepository
    from app.models.call_log import CallLog
except ImportError:
    HAS_DB = False

def normalize_name_transcript(text: str) -> str:
    """Lightweight name normalization for STT transcript."""
    if not text:
        return ""
    # Strip trailing punctuation
    t = text.strip().rstrip(".,!?")
    # Capitalize names if it starts with "i'm X" or "my name is X" or is just "akash"
    words = t.split()
    if len(words) == 1:
        # e.g., "akash" -> "Akash"
        return words[0].title()
    elif len(words) >= 2 and words[0].lower() in ("i'm", "this"):
        # e.g., "i'm akash" -> "I'm Akash"
        words[0] = "I'm"
        words[1:] = [w.title() for w in words[1:]]
        return " ".join(words)
    elif len(words) >= 3 and words[0].lower() == "my" and words[1].lower() == "name" and words[2].lower() == "is":
        # e.g., "my name is akash" -> "My name is Akash"
        words[3:] = [w.title() for w in words[3:]]
        return " ".join(words[:3] + words[3:])
    return t

def clean_speech_text(text: str) -> str:
    # Strip markdown bold/italic asterisks
    text = text.replace("**", "").replace("*", "")
    # Strip markdown headers (e.g. # Header)
    text = re.sub(r'#+\s+', '', text)
    # Strip backticks
    text = text.replace("`", "")
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    return text

class ConversationEngine:
    def __init__(self, db: Optional[Any] = None) -> None:
        self.db = db
        self.session_manager = SessionManager()
        self.llm_service = LLMService()
        self.prompt_service = PromptService(db) if db is not None else PromptService()
        self.rag_service = RAGService()
        self.call_log_repo = None

        if db is not None and HAS_DB:
            try:
                self.call_log_repo = CallLogRepository(db)
            except Exception:
                pass

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
        *args,
        **kwargs
    ) -> AsyncGenerator[Tuple[Optional[str], bool, bool], None]:
        """
        Streaming turn execution loop.
        Yields (text_token, should_hangup, should_transfer) progressively.
        """
        # Resolve customer_id, user_text, industry, language, agent_name from args/kwargs
        # Signature 1 (ai_cold_call): (customer_id, user_text)
        # Signature 2 (demo_cold_calling): (industry, language, agent_name, user_text)
        customer_id = kwargs.get("customer_id")
        user_text = kwargs.get("user_text")
        industry = kwargs.get("industry")
        language = kwargs.get("language")
        agent_name = kwargs.get("agent_name")
        
        if args:
            if len(args) == 2:
                # Signature 1: customer_id, user_text
                customer_id = args[0]
                user_text = args[1]
            elif len(args) == 4:
                # Signature 2: industry, language, agent_name, user_text
                industry = args[0]
                language = args[1]
                agent_name = args[2]
                user_text = args[3]

        # Mode A: Database-backed mode (ai_cold_call)
        if self.db is not None and customer_id is not None:
            history = await self.session_manager.get_message_history(call_id)
            state = await self.session_manager.get_session_state(call_id) or "greeting"

            # 1. Initialize session if empty
            if not history:
                compiled_prompt, _ = await self.prompt_service.build_prompt(
                    campaign_id,
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

        # Mode B: Database-less fallback mode (demo_cold_calling)
        else:
            state = await self.session_manager.get_session_state(call_id) or "GREETING"
            meta = await self.session_manager.get_session_metadata(call_id) or {}
            collected_info = meta.get("collected_info", {})

            # Retrieve dialogue history (keeping it clean of raw system templates)
            history = await self.session_manager.get_message_history(call_id)
            history_dialogue = [m for m in history if m["role"] in ("user", "assistant")]

            # Append user input if it's not the initial call start
            if user_text != "[CALL_START]":
                user_turn = {"role": "user", "content": user_text}
                history_dialogue.append(user_turn)
                await self.session_manager.append_message(call_id, user_turn)

            # Build dynamic prompt for the current turn based on state and variables
            compiled_prompt, _ = await self.prompt_service.build_prompt(
                campaign_id,
                industry,
                language,
                agent_name,
                state,
                collected_info,
                user_text
            )

            messages_to_send = [{"role": "system", "content": compiled_prompt}] + history_dialogue
            if user_text == "[CALL_START]":
                messages_to_send.append({"role": "user", "content": "[Please begin with your outbound greeting now.]"})

            # Stream completion and intercept tags dynamically
            should_hangup = False
            should_transfer = False
            full_text_accumulator = []
            tag_buffer = ""
            in_tag = False
            tool_calls_detected = None
            active_tools = [t for t in self._get_tools_schema() if t["function"]["name"] == "transfer_to_human"]

            async for text_chunk, t_calls in self.llm_service.generate_completion_stream(messages_to_send, active_tools):
                if t_calls:
                    tool_calls_detected = t_calls
                    break

                if text_chunk:
                    # Intercept tags starting with '['
                    if "[" in text_chunk:
                        in_tag = True
                        parts = text_chunk.split("[", 1)
                        if parts[0]:
                            clean_chunk = clean_speech_text(parts[0])
                            if clean_chunk:
                                full_text_accumulator.append(clean_chunk)
                                yield clean_chunk, False, False
                        tag_buffer += "[" + parts[1]
                    elif in_tag:
                        tag_buffer += text_chunk
                    else:
                        clean_chunk = clean_speech_text(text_chunk)
                        if clean_chunk:
                            full_text_accumulator.append(clean_chunk)
                            yield clean_chunk, False, False

            # Handle tool executions (e.g. transfer to human)
            if tool_calls_detected:
                for tool_call in tool_calls_detected:
                    func_name = tool_call.get("function", {}).get("name")
                    if func_name == "transfer_to_human":
                        should_transfer = True
                        logger.info(f"[CONV-CONTROLLER] Escalation tool called for session {call_id}")

            # Parse and extract next state and details from tag buffer
            next_state = None
            extracted_vars = {}

            state_match = re.search(r'\[STATE:\s*(\w+)\]', tag_buffer)
            if state_match:
                next_state = state_match.group(1).upper()

            extract_matches = re.findall(r'\[EXTRACT:\s*([^\]]+)\]', tag_buffer)
            for ext in extract_matches:
                pairs = re.findall(r'(\w+)\s*=\s*([^,\]]+)', ext)
                for k, v in pairs:
                    extracted_vars[k.strip().lower()] = v.strip()

            # Name Validation Guardrail: Verify extracted customer_name is plausible
            if "customer_name" in extracted_vars:
                name_val = extracted_vars["customer_name"].strip().title()
                invalid_names = {
                    "unknown", "none", "null", "undefined", "n/a", "user", "customer", 
                    "my gosh", "in the car", "my car", "gosh", "yes", "no", "hello", "hi", "ok", "okay"
                }
                if name_val.lower() in invalid_names or len(name_val) < 2 or not re.search(r'[A-Za-z\u0900-\u097F\u0C00-\u0C7F]', name_val):
                    logger.warning(f"[CONV-CONTROLLER] Rejected invalid extracted name '{name_val}'. Staying in IDENTITY_COLLECTION state.")
                    extracted_vars.pop("customer_name", None)
                    if state in ("GREETING", "IDENTITY_COLLECTION"):
                        next_state = "IDENTITY_COLLECTION"
                else:
                    extracted_vars["customer_name"] = name_val

            # Force transition to PURPOSE_OF_CALL if a name was successfully extracted
            if "customer_name" in extracted_vars and state in ("GREETING", "WAIT_FOR_NAME", "IDENTITY_COLLECTION"):
                next_state = "PURPOSE_OF_CALL"

            # Update controller state
            if next_state:
                await self.session_manager.update_session_state(call_id, next_state)
                logger.info(f"[CONV-CONTROLLER] Session {call_id} state transition: {state} -> {next_state}")
            
            # Update metadata details
            updated_info = dict(collected_info)
            if extracted_vars:
                updated_info.update(extracted_vars)
                await self.session_manager.update_session_metadata(call_id, {"collected_info": updated_info})

            # Append assistant speech response to conversation history
            full_assistant_text = "".join(full_text_accumulator).strip()
            if full_assistant_text:
                bot_turn = {"role": "assistant", "content": full_assistant_text}
                await self.session_manager.append_message(call_id, bot_turn)

            # Structured Stage Telemetry Logging
            logger.info(
                f"\n-------------------- [STAGE-3/4/5 TELEMETRY] --------------------\n"
                f"Session ID:      {call_id}\n"
                f"Current State:   {state}\n"
                f"Transcript:      '{user_text}'\n"
                f"Extracted Name:  {extracted_vars.get('customer_name', 'None')}\n"
                f"Extracted Vars:  {extracted_vars}\n"
                f"Next State:      {next_state or state}\n"
                f"Session Info:    {updated_info}\n"
                f"AI Spoke:        '{full_assistant_text}'\n"
                f"------------------------------------------------------------------"
            )

            # Evaluate completion / hangup conditions
            if next_state == "END_CALL" or state == "END_CALL":
                should_hangup = True
                logger.info(f"[CONV-CONTROLLER] Session {call_id} reached END_CALL → scheduling hangup.")

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
    ) -> Any:
        """Purge active Redis memory keys and flush completed conversation transcripts to PostgreSQL."""
        if not HAS_DB or self.call_log_repo is None:
            await self.session_manager.clear_session(call_id)
            return None

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
