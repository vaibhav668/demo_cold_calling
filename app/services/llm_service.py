import json
import httpx
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, AsyncGenerator
from app.core.config import settings
from app.core.logging import logger


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        """Abstract method to retrieve chat completion turns."""
        pass

    @abstractmethod
    async def generate_completion_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict[str, Any]]]], None]:
        """Abstract generator yielding text tokens or final tool calls."""
        pass

    async def _mock_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        last_user_message = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_message = m["content"].lower()
                break

        if "book" in last_user_message or "schedule" in last_user_message or "appointment" in last_user_message:
            if tools:
                return None, [
                    {
                        "id": "call_mock_book_123",
                        "type": "function",
                        "function": {
                            "name": "book_appointment",
                            "arguments": '{"date": "2026-08-01", "time": "14:00"}'
                        }
                    }
                ]

        if "human" in last_user_message or "operator" in last_user_message or "transfer" in last_user_message:
            if tools:
                return None, [
                    {
                        "id": "call_mock_transfer_123",
                        "type": "function",
                        "function": {
                            "name": "transfer_to_human",
                            "arguments": "{}"
                        }
                    }
                ]

        return "Hello! I can help you book an appointment or answer questions. How can I help?", None


class GeminiProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.LLM_MODEL
        self.url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    def _resolve_model(self) -> str:
        model = self.model
        if not model or "gemini" not in model.lower():
            return "gemini-2.0-flash"
        return model

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        if not self.api_key or self.api_key in ("test_gemini_key", "test_groq_key", "test_openai_key"):
            logger.warning("Gemini API key missing. Returning Mock completions...")
            return await self._mock_completion(messages, tools)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self._resolve_model(),
            "messages": messages
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(self.url, headers=headers, json=payload)
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"Gemini error code {response.status_code}: {response.text[:200]}",
                    request=response.request,
                    response=response
                )

            data = response.json()
            choice = data["choices"][0]["message"]
            return choice.get("content"), choice.get("tool_calls")

    async def generate_completion_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict[str, Any]]]], None]:
        if not self.api_key or self.api_key in ("test_gemini_key", "test_groq_key", "test_openai_key"):
            content, t_calls = await self._mock_completion(messages, tools)
            yield content, t_calls
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self._resolve_model(),
            "messages": messages,
            "stream": True
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                async with client.stream("POST", self.url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        logger.warning(f"[Gemini Stream] Error {response.status_code}: {error_body.decode()[:200]}. Falling back to non-stream.")
                        content, t_calls = await self.generate_completion(messages, tools)
                        yield content, t_calls
                        return

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            choices = chunk_data.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})

                            text_delta = delta.get("content")
                            if text_delta:
                                yield text_delta, None

                            t_deltas = delta.get("tool_calls")
                            if t_deltas:
                                for td in t_deltas:
                                    idx = td.get("index", 0)
                                    if idx not in tool_calls_accumulator:
                                        tool_calls_accumulator[idx] = {
                                            "id": td.get("id", ""),
                                            "type": td.get("type", "function"),
                                            "function": {"name": "", "arguments": ""}
                                        }
                                    if td.get("id"):
                                        tool_calls_accumulator[idx]["id"] = td["id"]
                                    fn = td.get("function", {})
                                    if fn.get("name"):
                                        tool_calls_accumulator[idx]["function"]["name"] += fn["name"]
                                    if fn.get("arguments"):
                                        tool_calls_accumulator[idx]["function"]["arguments"] += fn["arguments"]

                        except Exception as parse_err:
                            logger.debug(f"[Gemini Stream] Parse line error: {parse_err}")

        except Exception as e:
            logger.warning(f"[Gemini Stream] Streaming failed: {e}. Falling back to non-stream.")
            content, t_calls = await self.generate_completion(messages, tools)
            yield content, t_calls
            return

        if tool_calls_accumulator:
            final_tools = [v for k, v in sorted(tool_calls_accumulator.items())]
            yield None, final_tools


class GroqProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.LLM_MODEL
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def _resolve_model(self) -> str:
        model = self.model
        if not model or "gemini" in model.lower() or model in ("qwen3-instruct", "qwen-2.5-32b"):
            return "llama-3.3-70b-versatile"
        return model

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        if not self.api_key or self.api_key in ("test_groq_key", "test_openai_key"):
            logger.warning("Groq API key missing. Returning Mock completions...")
            return await self._mock_completion(messages, tools)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self._resolve_model(),
            "messages": messages
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(self.url, headers=headers, json=payload)
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"Groq error code {response.status_code}",
                    request=response.request,
                    response=response
                )

            data = response.json()
            choice = data["choices"][0]["message"]
            return choice.get("content"), choice.get("tool_calls")

    async def generate_completion_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict[str, Any]]]], None]:
        if not self.api_key or self.api_key in ("test_groq_key", "test_openai_key"):
            content, t_calls = await self._mock_completion(messages, tools)
            yield content, t_calls
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self._resolve_model(),
            "messages": messages,
            "stream": True
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}

        async with httpx.AsyncClient(timeout=15.0) as client:
            async with client.stream("POST", self.url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise httpx.HTTPStatusError(
                        f"Groq stream error code {response.status_code}: {error_body.decode()}",
                        request=response.request,
                        response=response
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        choices = chunk_data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        # Check for text content
                        text_delta = delta.get("content")
                        if text_delta:
                            yield text_delta, None

                        # Check for tool calls delta
                        t_deltas = delta.get("tool_calls")
                        if t_deltas:
                            for td in t_deltas:
                                idx = td.get("index", 0)
                                if idx not in tool_calls_accumulator:
                                    tool_calls_accumulator[idx] = {
                                        "id": td.get("id", ""),
                                        "type": td.get("type", "function"),
                                        "function": {"name": "", "arguments": ""}
                                    }
                                if td.get("id"):
                                    tool_calls_accumulator[idx]["id"] = td["id"]
                                fn = td.get("function", {})
                                if fn.get("name"):
                                    tool_calls_accumulator[idx]["function"]["name"] += fn["name"]
                                if fn.get("arguments"):
                                    tool_calls_accumulator[idx]["function"]["arguments"] += fn["arguments"]

                    except Exception as parse_err:
                        logger.debug(f"[Groq Stream] Parse line error: {parse_err}")

        # If tool calls were accumulated, yield them at end of stream
        if tool_calls_accumulator:
            final_tools = [v for k, v in sorted(tool_calls_accumulator.items())]
            yield None, final_tools


class LLMManager:
    def __init__(self) -> None:
        provider_type = settings.LLM_PROVIDER.lower() if settings.LLM_PROVIDER else "gemini"
        if provider_type == "groq":
            self.primary = GroqProvider()
            self.fallback = GeminiProvider()
        else:
            # Default main provider: Gemini API (gemini-2.0-flash)
            self.primary = GeminiProvider()
            self.fallback = GroqProvider()

    def _get_fallback_message(self, messages: List[Dict[str, str]]) -> str:
        for m in messages:
            content = m.get("content", "")
            if "Hindi" in content or "हिंदी" in content or any('\u0900' <= c <= '\u097F' for c in content):
                return "क्षमा करें, कनेक्शन में समस्या आ रही है। क्या आप दोहरा सकते हैं?"
            if "Telugu" in content or "తెలుగు" in content or any('\u0c00' <= c <= '\u0c7f' for c in content):
                return "క్షమించండి, కనెక్షన్ సమస్య ఉంది. దయచేసి మళ్లీ చెప్పగలరా?"
        return "I am having trouble connecting right now. Could you repeat that?"

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        try:
            return await self.primary.generate_completion(messages, tools)
        except Exception as e:
            logger.warning(f"Primary LLM Provider ({self.primary.__class__.__name__}) failed: {e}. Switching to Fallback ({self.fallback.__class__.__name__})...")
            try:
                return await self.fallback.generate_completion(messages, tools)
            except Exception as fe:
                logger.error(f"Fallback LLM Provider failed: {fe}")
                return self._get_fallback_message(messages), None

    async def generate_completion_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict[str, Any]]]], None]:
        try:
            async for chunk in self.primary.generate_completion_stream(messages, tools):
                yield chunk
        except Exception as e:
            logger.warning(f"[LLM Stream] Primary stream ({self.primary.__class__.__name__}) failed: {e}. Falling back to secondary...")
            try:
                async for chunk in self.fallback.generate_completion_stream(messages, tools):
                    yield chunk
            except Exception as fe:
                logger.error(f"[LLM Stream] Fallback failed: {fe}")
                yield self._get_fallback_message(messages), None


LLMService = LLMManager
