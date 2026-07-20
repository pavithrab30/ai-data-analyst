"""
NVIDIA NIM LLM service.

Uses direct httpx calls to the NVIDIA NIM OpenAI-compatible REST API.
No third-party LLM SDKs — just httpx for reliability and timeout control.

All tools import `llm_service` from this module:
    from services.llm_service import llm_service
"""

from __future__ import annotations

import json
import time
from typing import Generator, Optional

import httpx

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_TIMEOUT = httpx.Timeout(180.0, connect=15.0, read=180.0, write=30.0)


class LLMServiceError(Exception):
    """Raised on unrecoverable LLM API errors."""


class LLMService:
    """
    NVIDIA NIM inference client.

    Wraps the OpenAI-compatible NVIDIA NIM REST endpoint using
    direct httpx calls for full timeout and error control.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or settings.nvidia_api_key
        self._model = model or settings.nvidia_model

        if not self._api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Get a free key at https://build.nvidia.com and add it to .env"
            )

        logger.info("LLMService initialized", model=self._model)

    # ── Public interface ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature_override: Optional[float] = None,
    ) -> str:
        """
        Single-turn text generation.

        Args:
            prompt: User prompt.
            system_instruction: Optional system message prepended to conversation.
            temperature_override: Override default temperature for this call.

        Returns:
            Generated text string.
        """
        messages = self._build_messages(prompt, system_instruction)
        return self._call(messages, temperature_override)

    def generate_with_history(
        self,
        messages: list[dict[str, str]],
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Multi-turn chat generation using conversation history.

        Args:
            messages: List of {"role": "user"|"assistant"|"model", "content": str}.
            system_instruction: Optional system message.

        Returns:
            Generated text string.
        """
        formatted: list[dict[str, str]] = []
        if system_instruction:
            formatted.append({"role": "system", "content": system_instruction})
        for m in messages:
            # Normalise "model" role (Gemini legacy) to "assistant"
            role = "assistant" if m["role"] in ("assistant", "model") else "user"
            formatted.append({"role": role, "content": m["content"]})

        return self._call(formatted)

    def stream_generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming generation — yields text chunks as they arrive.

        Args:
            prompt: User prompt.
            system_instruction: Optional system message.

        Yields:
            Text delta strings.
        """
        messages = self._build_messages(prompt, system_instruction)
        yield from self._stream_call(messages)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_messages(
        self,
        prompt: str,
        system_instruction: Optional[str],
    ) -> list[dict[str, str]]:
        """Build the messages list for a single-turn call."""
        msgs: list[dict[str, str]] = []
        if system_instruction:
            msgs.append({"role": "system", "content": system_instruction})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _call(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        """POST to NVIDIA NIM and return the response text."""
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "stream": False,
        }

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.post(
                    _BASE_URL,
                    headers=self._headers(),
                    content=json.dumps(payload),
                )
                resp.raise_for_status()
                data = resp.json()

            elapsed = (time.perf_counter() - start) * 1000
            text: str = data["choices"][0]["message"]["content"] or ""
            logger.debug(
                "NVIDIA NIM response",
                model=self._model,
                elapsed_ms=round(elapsed, 2),
                response_length=len(text),
            )
            return text

        except httpx.TimeoutException as exc:
            logger.error("NVIDIA NIM request timed out", error=str(exc))
            raise LLMServiceError(
                "The NVIDIA NIM request timed out. "
                "Try a smaller model or reduce MAX_TOKENS in .env."
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                "NVIDIA NIM HTTP error",
                status=exc.response.status_code,
                body=exc.response.text[:300],
            )
            raise LLMServiceError(
                f"NVIDIA NIM returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except Exception as exc:
            logger.error("NVIDIA NIM unexpected error", error=str(exc))
            raise LLMServiceError(f"NVIDIA NIM error: {exc}") from exc

    def _stream_call(
        self,
        messages: list[dict[str, str]],
    ) -> Generator[str, None, None]:
        """POST with stream=True and yield text deltas."""
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "stream": True,
        }

        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                with client.stream(
                    "POST",
                    _BASE_URL,
                    headers={**self._headers(), "Accept": "text/event-stream"},
                    content=json.dumps(payload),
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                chunk = json.loads(line[6:])
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield delta
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
        except Exception as exc:
            logger.error("NVIDIA NIM stream error", error=str(exc))
            raise LLMServiceError(f"Streaming error: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
llm_service = LLMService()
