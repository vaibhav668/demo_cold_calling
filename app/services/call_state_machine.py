import asyncio
from enum import Enum, auto
from app.core.logging import logger

class CallState(Enum):
    CONNECTED = auto()
    AI_SPEAKING = auto()
    WAITING_FOR_CUSTOMER = auto()
    CUSTOMER_SPEAKING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    GENERATING_RESPONSE = auto()
    CALL_COMPLETED = auto()
    ERROR = auto()

_VALID_TRANSITIONS: dict[CallState, set[CallState]] = {
    CallState.CONNECTED: {
        CallState.THINKING,
        CallState.GENERATING_RESPONSE,
        CallState.AI_SPEAKING,
        CallState.WAITING_FOR_CUSTOMER,
        CallState.ERROR,
    },
    CallState.AI_SPEAKING: {
        CallState.WAITING_FOR_CUSTOMER,
        CallState.CUSTOMER_SPEAKING,
        CallState.CALL_COMPLETED,
        CallState.ERROR,
    },
    CallState.WAITING_FOR_CUSTOMER: {
        CallState.CUSTOMER_SPEAKING,
        CallState.THINKING,
        CallState.CALL_COMPLETED,
        CallState.ERROR,
    },
    CallState.CUSTOMER_SPEAKING: {
        CallState.TRANSCRIBING,
        CallState.WAITING_FOR_CUSTOMER,
        CallState.ERROR,
    },
    CallState.TRANSCRIBING: {
        CallState.THINKING,
        CallState.WAITING_FOR_CUSTOMER,
        CallState.ERROR,
    },
    CallState.THINKING: {
        CallState.GENERATING_RESPONSE,
        CallState.THINKING,
        CallState.WAITING_FOR_CUSTOMER,
        CallState.ERROR,
    },
    CallState.GENERATING_RESPONSE: {
        CallState.AI_SPEAKING,
        CallState.WAITING_FOR_CUSTOMER,
        CallState.CALL_COMPLETED,
        CallState.ERROR,
    },
    CallState.CALL_COMPLETED: set(),
    CallState.ERROR: set(),
}

class CallStateMachine:
    def __init__(self, call_uuid: str) -> None:
        self.call_uuid = call_uuid
        self._state = CallState.CONNECTED
        self._lock = asyncio.Lock()
        self.ai_speech_start_time = 0.0
        self.waiting_start_time = 0.0
        logger.info(f"[STATE] {call_uuid} → CONNECTED")

    @property
    def state(self) -> CallState:
        return self._state

    async def transition(self, new_state: CallState) -> bool:
        async with self._lock:
            allowed = _VALID_TRANSITIONS.get(self._state, set())
            if new_state not in allowed:
                logger.warning(
                    f"[STATE] {self.call_uuid} INVALID transition "
                    f"{self._state.name} → {new_state.name} (ignored)"
                )
                return False

            old_state = self._state
            self._state = new_state
            
            loop_time = asyncio.get_event_loop().time()
            if new_state == CallState.AI_SPEAKING:
                self.ai_speech_start_time = 999999999.0
            elif new_state == CallState.WAITING_FOR_CUSTOMER:
                self.waiting_start_time = loop_time

            logger.info(
                f"[STATE] {self.call_uuid} {old_state.name} → {new_state.name}"
            )
            return True

    def force(self, new_state: CallState) -> None:
        self._state = new_state

    def is_terminal(self) -> bool:
        return self._state in (CallState.CALL_COMPLETED, CallState.ERROR)

    def is_ai_speaking(self) -> bool:
        return self._state == CallState.AI_SPEAKING

    def is_waiting(self) -> bool:
        return self._state == CallState.WAITING_FOR_CUSTOMER

    def is_customer_turn(self) -> bool:
        return self._state in (
            CallState.CUSTOMER_SPEAKING,
            CallState.TRANSCRIBING,
            CallState.THINKING,
            CallState.GENERATING_RESPONSE,
        )
