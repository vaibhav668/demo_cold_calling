import uuid
import json
import re
from typing import Tuple, List, Dict, Any, Optional, AsyncGenerator
from app.services.session_manager import SessionManager
from app.services.llm_service import LLMService
from app.services.prompt_service import PromptService
from app.services.rag_service import RAGService
from app.core.logging import logger

def clean_speech_text(text: str) -> str:
    """Strip markdown formatting, headers, HTML tags, and formatting symbols to prevent reading them."""
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
    def __init__(self) -> None:
        self.session_manager = SessionManager()
        self.llm_service = LLMService()
        self.prompt_service = PromptService()
        self.rag_service = RAGService()

    def _get_tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "transfer_to_human",
                    "description": "Transfer the call to a human operator or sales representative.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

    async def process_turn_stream(
        self,
        call_id: str,
        campaign_id: uuid.UUID,
        industry: str,
        language: str,
        agent_name: str,
        user_text: str
    ) -> AsyncGenerator[Tuple[Optional[str], bool, bool], None]:
        """
        State-driven conversation turn execution loop.
        Yields (text_token, should_hangup, should_transfer) progressively.
        """
        # 1. Retrieve current state and collected info from session manager
        state = await self.session_manager.get_session_state(call_id) or "GREETING"
        meta = await self.session_manager.get_session_metadata(call_id) or {}
        collected_info = meta.get("collected_info", {})

        # Retrieve dialogue history (keeping it clean of raw system templates)
        history = await self.session_manager.get_message_history(call_id)
        history_dialogue = [m for m in history if m["role"] in ("user", "assistant")]

        # 2. Append user input if it's not the initial call start
        if user_text != "[CALL_START]":
            user_turn = {"role": "user", "content": user_text}
            history_dialogue.append(user_turn)
            await self.session_manager.append_message(call_id, user_turn)

        # 3. Build dynamic prompt for the current turn based on state and variables
        compiled_prompt, _ = await self.prompt_service.build_prompt(
            campaign_id=campaign_id,
            industry=industry,
            language=language,
            agent_name=agent_name,
            current_state=state,
            collected_info=collected_info,
            rag_query=user_text
        )

        messages_to_send = [{"role": "system", "content": compiled_prompt}] + history_dialogue
        if user_text == "[CALL_START]":
            messages_to_send.append({"role": "user", "content": "[Please begin with your outbound greeting now.]"})

        # 4. Stream completion and intercept tags dynamically
        should_hangup = False
        should_transfer = False
        full_text_accumulator = []
        tag_buffer = ""
        in_tag = False
        tool_calls_detected = None
        active_tools = self._get_tools_schema()

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

        # 5. Handle tool executions (e.g. transfer to human)
        if tool_calls_detected:
            for tool_call in tool_calls_detected:
                func_name = tool_call.get("function", {}).get("name")
                if func_name == "transfer_to_human":
                    should_transfer = True
                    logger.info(f"[CONV-CONTROLLER] Escalation tool called for session {call_id}")

        # 6. Parse and extract next state and details from tag buffer
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

        # 7. Evaluate completion / hangup conditions
        # Deterministic: if LLM tagged the next state as END_CALL, hang up immediately.
        # No fragile word-matching — state transition is the single source of truth.
        if next_state == "END_CALL" or state == "END_CALL":
            should_hangup = True
            logger.info(f"[CONV-CONTROLLER] Session {call_id} reached END_CALL → scheduling hangup.")

        yield None, should_hangup, should_transfer
