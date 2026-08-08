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


class GroqProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.LLM_MODEL
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def _resolve_model(self) -> str:
        model = self.model
        if model == "qwen3-instruct":
            return "qwen-2.5-32b"
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


class OpenRouterProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.api_key = settings.OPENROUTER_API_KEY
        self.model = settings.LLM_MODEL
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def _resolve_model(self) -> str:
        model = self.model
        if "llama-3.1-8b" in model:
            return "meta-llama/llama-3.1-8b-instruct"
        if model == "qwen3-instruct":
            return "qwen/qwen-2.5-72b-instruct"
        return model

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        if not self.api_key or self.api_key in ("test_openrouter_key", "test_openai_key"):
            return await self._mock_completion(messages, tools)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://voice-agent-api.onrender.com",
            "X-Title": "VoiceAgent.AI"
        }
        payload = {"model": self._resolve_model(), "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(self.url, headers=headers, json=payload)
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"OpenRouter error code {response.status_code}",
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
        """Real SSE streaming implementation for OpenRouter."""
        if not self.api_key or self.api_key in ("test_openrouter_key", "test_openai_key"):
            content, t_calls = await self.generate_completion(messages, tools)
            yield content, t_calls
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://voice-agent-api.onrender.com",
            "X-Title": "VoiceAgent.AI"
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
                        logger.warning(f"[OpenRouter Stream] Error {response.status_code}: {error_body.decode()[:200]}. Falling back to non-stream.")
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
                            logger.debug(f"[OpenRouter Stream] Parse line error: {parse_err}")

        except Exception as e:
            logger.warning(f"[OpenRouter Stream] Streaming failed: {e}. Falling back to non-stream.")
            content, t_calls = await self.generate_completion(messages, tools)
            yield content, t_calls
            return

        if tool_calls_accumulator:
            final_tools = [v for k, v in sorted(tool_calls_accumulator.items())]
            yield None, final_tools


class LLMManager:
    def __init__(self) -> None:
        provider_type = settings.LLM_PROVIDER.lower() if settings.LLM_PROVIDER else "groq"
        if provider_type == "qwen":
            # Groq decommissioned qwen-2.5-32b; route Qwen models directly to OpenRouter.
            self.primary = OpenRouterProvider()
            self.fallback = GroqProvider()
        elif provider_type == "groq":
            self.primary = GroqProvider()
            self.fallback = OpenRouterProvider()
        elif provider_type == "openrouter":
            self.primary = OpenRouterProvider()
            self.fallback = GroqProvider()
        else:
            self.primary = GroqProvider()
            self.fallback = OpenRouterProvider()

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        try:
            return await self.primary.generate_completion(messages, tools)
        except Exception as e:
            logger.warning(f"Primary LLM Provider failed: {e}. Switching to Fallback Provider...")
            try:
                return await self.fallback.generate_completion(messages, tools)
            except Exception as fe:
                logger.error(f"Fallback LLM Provider failed: {fe}")
                return "I am having trouble connecting right now. Can you repeat that?", None

    async def generate_completion_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict[str, Any]]]], None]:
        try:
            async for chunk in self.primary.generate_completion_stream(messages, tools):
                yield chunk
        except Exception as e:
            logger.warning(f"[LLM Stream] Primary stream failed: {e}. Falling back to secondary...")
            try:
                async for chunk in self.fallback.generate_completion_stream(messages, tools):
                    yield chunk
            except Exception as fe:
                logger.error(f"[LLM Stream] Fallback failed: {fe}")
                yield "I am having trouble connecting right now. Could you repeat that?", None


LLMService = LLMManager
