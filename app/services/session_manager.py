import gc
from typing import List, Dict, Optional
from app.core.logging import logger

_in_memory_states: Dict[str, str] = {}
_in_memory_messages: Dict[str, List[Dict[str, str]]] = {}
_in_memory_metadata: Dict[str, Dict] = {}

class SessionManager:
    async def get_session_state(self, call_id: str) -> Optional[str]:
        return _in_memory_states.get(call_id)

    async def update_session_state(self, call_id: str, state: str) -> None:
        _in_memory_states[call_id] = state

    async def get_session_metadata(self, call_id: str) -> Optional[Dict]:
        return _in_memory_metadata.get(call_id)

    async def update_session_metadata(self, call_id: str, metadata: Dict) -> None:
        if call_id not in _in_memory_metadata:
            _in_memory_metadata[call_id] = {}
        _in_memory_metadata[call_id].update(metadata)

    async def get_message_history(self, call_id: str) -> List[Dict[str, str]]:
        return _in_memory_messages.get(call_id, [])

    async def clear_message_history(self, call_id: str) -> None:
        _in_memory_messages.pop(call_id, None)

    async def append_message(self, call_id: str, message: Dict[str, str]) -> None:
        if call_id not in _in_memory_messages:
            _in_memory_messages[call_id] = []
        _in_memory_messages[call_id].append(message)

    async def clear_session(self, call_id: str) -> None:
        _in_memory_states.pop(call_id, None)
        _in_memory_messages.pop(call_id, None)
        _in_memory_metadata.pop(call_id, None)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info(f"[SessionManager] Purged local session cache for {call_id} and ran garbage collection.")


class VoiceSession:
    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def customer_name(self) -> Optional[str]:
        meta = _in_memory_metadata.get(self.session_id)
        if meta:
            return meta.get("customer_name")
        return None

    @customer_name.setter
    def customer_name(self, val: Optional[str]) -> None:
        if self.session_id not in _in_memory_metadata:
            _in_memory_metadata[self.session_id] = {}
        _in_memory_metadata[self.session_id]["customer_name"] = val

