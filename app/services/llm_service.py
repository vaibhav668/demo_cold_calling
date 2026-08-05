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
        pass

    @abstractmethod
    async def generate_completion_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict[str, Any]]]], None]:
        pass

    async def _mock_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        last_user_message = ""
        is_hospital = False
        for m in messages:
            if m["role"] == "system" and "CityCare" in m.get("content", ""):
                is_hospital = True
                break

        for m in reversed(messages):
            if m["role"] == "user":
                last_user_message = m["content"].lower()
                break

        # Check for tool queries
        if "book" in last_user_message or "schedule" in last_user_message or "appointment" in last_user_message or "visit" in last_user_message:
            if tools:
                return None, [
                    {
                        "id": "call_mock_book_123",
                        "type": "function",
                        "function": {
                            "name": "book_appointment",
                            "arguments": '{"date": "2026-08-10", "time": "11:00"}'
                        }
                    }
                ]

        if "questions" in last_user_message or "parking" in last_user_message or "location" in last_user_message or "cancel" in last_user_message or "amenit" in last_user_message or "price" in last_user_message:
            if tools:
                return None, [
                    {
                        "id": "call_mock_kb_123",
                        "type": "function",
                        "function": {
                            "name": "lookup_knowledge",
                            "arguments": f'{{"query": "{last_user_message}"}}'
                        }
                    }
                ]

        if is_hospital:
            return "Sure! I can help you confirm your cardiology appointment or reschedule it if needed. Would you like me to book it for tomorrow?", None
        else:
            return "Orchard Heights is a premium luxury housing project located in Gachibowli. We offer 2 BHK and 3 BHK homes starting from 80 Lakhs. Would you like to schedule a site visit?", None

class GroqProvider(BaseLLMProvider):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.LLM_MODEL
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.client = client

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        if not self.api_key or self.api_key in ("test_groq_key", "test_openai_key"):
            logger.warning("[LLM] Groq API key missing. Using Mock completions...")
            return await self._mock_completion(messages, tools)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = await self.client.post(self.url, headers=headers, json=payload)
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
            "model": self.model,
            "messages": messages,
            "stream": True
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}

        async with self.client.stream("POST", self.url, headers=headers, json=payload) as response:
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
                    logger.debug(f"[Groq Stream] Parse line error: {parse_err}")

        if tool_calls_accumulator:
            final_tools = [v for k, v in sorted(tool_calls_accumulator.items())]
            yield None, final_tools

class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.api_key = settings.OPENROUTER_API_KEY
        self.model = settings.LLM_MODEL
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.client = client

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
            "HTTP-Referer": "https://voice-agent-demo.onrender.com",
            "X-Title": "VoiceAgentDemo.AI"
        }
        model = self.model
        if "llama-3.1-8b" in model:
            model = "meta-llama/llama-3.1-8b-instruct"

        payload = {"model": model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = await self.client.post(self.url, headers=headers, json=payload)
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
        content, t_calls = await self.generate_completion(messages, tools)
        yield content, t_calls

class LLMManager:
    _shared_client: Optional[httpx.AsyncClient] = None

    def __init__(self) -> None:
        if LLMManager._shared_client is None:
            # Shared HTTP Client with connection pool keep-alive
            logger.info("[LLM] Initializing shared httpx.AsyncClient connection pool...")
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            LLMManager._shared_client = httpx.AsyncClient(limits=limits, timeout=15.0)

        self.primary = GroqProvider(LLMManager._shared_client)
        self.fallback = OpenRouterProvider(LLMManager._shared_client)

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        try:
            return await self.primary.generate_completion(messages, tools)
        except Exception as e:
            logger.warning(f"Primary LLM Provider (Groq) failed: {e}. Switching to Fallback...")
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
            logger.warning(f"[LLM Stream] Primary Groq stream failed: {e}. Switching to Fallback...")
            try:
                async for chunk in self.fallback.generate_completion_stream(messages, tools):
                    yield chunk
            except Exception as fe:
                logger.error(f"[LLM Stream] Fallback failed: {fe}")
                yield "I am having trouble connecting right now. Could you repeat that?", None

LLMService = LLMManager
