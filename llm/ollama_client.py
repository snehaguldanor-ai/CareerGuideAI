"""Thin wrapper around the Ollama Python client with friendly error mapping."""
from __future__ import annotations

from typing import Iterator

import httpx
import ollama

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_HOST = "http://localhost:11434"


class LLMError(Exception):
    """Generic LLM failure."""


class OllamaUnavailableError(LLMError):
    """Ollama server is not running / not reachable."""


class ModelNotFoundError(LLMError):
    """The requested model is not pulled locally."""


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: float = 600.0) -> None:
        self.model = model
        self.host = host
        self.client = ollama.Client(host=host, timeout=timeout)

    # ------------------------------------------------------------------ status
    def _installed_models(self) -> list[str]:
        response = self.client.list()
        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models", [])
        names: list[str] = []
        for m in models or []:
            if isinstance(m, dict):
                name = m.get("model") or m.get("name")
            else:
                name = getattr(m, "model", None) or getattr(m, "name", None)
            if name:
                names.append(str(name))
        return names

    def check(self) -> tuple[bool, str]:
        """Return (ok, message) describing server + model availability."""
        try:
            names = self._installed_models()
        except Exception as exc:  # noqa: BLE001
            return False, (
                f"Ollama is not reachable at {self.host}. Start the Ollama app (or run `ollama serve`) "
                f"and try again. ({exc.__class__.__name__})"
            )
        wanted = {self.model, f"{self.model}:latest"}
        if not any(n in wanted for n in names):
            return False, f"Model '{self.model}' is not installed. Run `ollama pull {self.model}` in a terminal."
        return True, f"Ollama is running and '{self.model}' is ready."

    # ------------------------------------------------------------------ errors
    def _map_error(self, exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        message = str(exc)
        lowered = message.lower()
        if isinstance(exc, ollama.ResponseError):
            if getattr(exc, "status_code", None) == 404 or "not found" in lowered:
                return ModelNotFoundError(
                    f"Model '{self.model}' was not found. Run `ollama pull {self.model}` and retry."
                )
            return LLMError(f"Ollama returned an error: {message}")
        if isinstance(exc, (httpx.ConnectError, ConnectionError)) or "refused" in lowered or "connect" in lowered:
            return OllamaUnavailableError(
                f"Cannot connect to Ollama at {self.host}. Start Ollama (open the app or run `ollama serve`)."
            )
        if isinstance(exc, httpx.TimeoutException):
            return LLMError("The model took too long to respond. Try again or use a smaller input.")
        return LLMError(f"Unexpected LLM error: {message}")

    # ------------------------------------------------------------------ calls
    def _messages(self, prompt: str, system: str) -> list[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        options: dict = {"temperature": temperature, "num_ctx": 4096}
        if max_tokens:
            options["num_predict"] = max_tokens
        try:
            response = self.client.chat(
                model=self.model,
                messages=self._messages(prompt, system),
                options=options,
                format="json" if json_mode else None,
            )
            content = response["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise self._map_error(exc) from exc
        if not content or not content.strip():
            raise LLMError("The model returned an empty answer. Please try again.")
        return content.strip()

    def stream(self, prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int | None = 800) -> Iterator[str]:
        options: dict = {"temperature": temperature, "num_ctx": 4096}
        if max_tokens:
            options["num_predict"] = max_tokens
        try:
            for part in self.client.chat(
                model=self.model,
                messages=self._messages(prompt, system),
                options=options,
                stream=True,
            ):
                piece = part["message"]["content"]
                if piece:
                    yield piece
        except Exception as exc:  # noqa: BLE001
            raise self._map_error(exc) from exc
