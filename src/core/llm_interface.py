"""
Unified LLM interface supporting runtime routing across multiple backends.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple

from src.config import CONFIG, LLMBackend


class LLMInterface:
    """Unified interface for multiple LLM backends with per-function routing."""

    DEFAULT_MODELS = {
        LLMBackend.OPENAI: "gpt-4o-mini",
        LLMBackend.GEMINI: "gemini-1.5-flash",
        LLMBackend.OLLAMA: "mistral:latest",
        LLMBackend.LM_STUDIO: "local-model",
        LLMBackend.ANTHROPIC: "claude-3-5-sonnet-20241022",
    }

    DEFAULT_BASE_URLS = {
        LLMBackend.OPENAI: "https://api.openai.com/v1",
        LLMBackend.GEMINI: "https://generativelanguage.googleapis.com",
        LLMBackend.OLLAMA: "http://localhost:11434/v1",
        LLMBackend.LM_STUDIO: "http://localhost:1234/v1",
        LLMBackend.ANTHROPIC: "https://api.anthropic.com",
    }

    def __init__(self):
        self.config = CONFIG.llm
        self.default_backend = self._normalize_backend(self.config.backend, fallback=LLMBackend.OLLAMA)
        self.default_model_name = self.config.model_name or self.DEFAULT_MODELS[self.default_backend]
        self.default_api_key = self.config.api_key or ""
        self.default_base_url = self.config.base_url or self.DEFAULT_BASE_URLS[self.default_backend]
        self.temperature = self.config.temperature
        self.max_tokens = self.config.max_tokens
        self.timeout = self.config.timeout
        self.last_error = ""

        # Function-specific routing map, e.g. "summarization" -> {backend/model/...}
        self._routing: Dict[str, Dict[str, Any]] = {}
        self._client_cache: Dict[Tuple[str, str, str], Any] = {}

    @property
    def backend(self) -> LLMBackend:
        """Backwards-compatible accessor for current default backend."""
        return self.default_backend

    def _normalize_backend(self, backend: Any, fallback: LLMBackend) -> LLMBackend:
        """Normalize backend values from enum/string."""
        if isinstance(backend, LLMBackend):
            return backend
        if backend is None:
            return fallback

        backend_value = str(backend).strip().lower()
        for candidate in LLMBackend:
            if candidate.value == backend_value:
                return candidate
        return fallback

    def _normalize_route(self, route: Dict[str, Any], fallback_route: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize route payload with defaults."""
        backend = self._normalize_backend(route.get("backend"), fallback=fallback_route["backend"])
        model_name = route.get("model_name") or route.get("model") or fallback_route.get("model_name") or self.DEFAULT_MODELS[backend]
        api_key = route.get("api_key", fallback_route.get("api_key", ""))
        base_url = route.get("base_url") or fallback_route.get("base_url") or self.DEFAULT_BASE_URLS[backend]

        return {
            "backend": backend,
            "model_name": model_name,
            "api_key": api_key or "",
            "base_url": base_url or self.DEFAULT_BASE_URLS[backend],
        }

    def apply_runtime_settings(self, settings: Optional[Dict[str, Any]]) -> None:
        """Apply runtime backend/model settings and optional per-function routing."""
        if not settings:
            return

        base_route = {
            "backend": self.default_backend,
            "model_name": self.default_model_name,
            "api_key": self.default_api_key,
            "base_url": self.default_base_url,
        }
        normalized_default = self._normalize_route(settings, fallback_route=base_route)
        self.default_backend = normalized_default["backend"]
        self.default_model_name = normalized_default["model_name"]
        self.default_api_key = normalized_default["api_key"]
        self.default_base_url = normalized_default["base_url"]

        if "temperature" in settings:
            self.temperature = float(settings["temperature"])
        if "max_tokens" in settings:
            self.max_tokens = int(settings["max_tokens"])
        if "timeout" in settings:
            self.timeout = int(settings["timeout"])

        routing = settings.get("routing", {}) or {}
        new_routing: Dict[str, Dict[str, Any]] = {}
        for task_type, route in routing.items():
            if not isinstance(route, dict):
                continue
            new_routing[str(task_type).strip().lower()] = self._normalize_route(route, fallback_route=normalized_default)

        if new_routing:
            self._routing = new_routing

    def _get_client(self, backend: LLMBackend, api_key: str, base_url: str) -> Any:
        """Create or fetch cached client for backend."""
        cache_key = (backend.value, base_url or "", api_key or "")

        if backend == LLMBackend.GEMINI:
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise ImportError("google-generativeai package required") from e
            if not api_key:
                raise ValueError("Gemini requires an API key")
            genai.configure(api_key=api_key)
            return genai

        if cache_key in self._client_cache:
            return self._client_cache[cache_key]

        if backend in (LLMBackend.OPENAI, LLMBackend.OLLAMA, LLMBackend.LM_STUDIO):
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("openai package required") from e

            resolved_base_url = base_url or self.DEFAULT_BASE_URLS[backend]
            if backend == LLMBackend.OPENAI:
                resolved_api_key = api_key
                if not resolved_api_key:
                    raise ValueError("OpenAI backend requires an API key")
            elif backend == LLMBackend.OLLAMA:
                resolved_api_key = api_key or "ollama"
            else:
                resolved_api_key = api_key or "lm-studio"

            client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
            self._client_cache[cache_key] = client
            return client

        if backend == LLMBackend.ANTHROPIC:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise ImportError("anthropic package required") from e
            if not api_key:
                raise ValueError("Anthropic backend requires an API key")
            client = Anthropic(api_key=api_key)
            self._client_cache[cache_key] = client
            return client

        raise ValueError(f"Unsupported backend: {backend}")

    def _resolve_route(self, task_type: str = "default", route_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resolve backend/model route for a generation request."""
        resolved = {
            "backend": self.default_backend,
            "model_name": self.default_model_name or self.DEFAULT_MODELS[self.default_backend],
            "api_key": self.default_api_key,
            "base_url": self.default_base_url,
        }

        task_key = (task_type or "default").strip().lower()
        if task_key in self._routing:
            resolved = self._normalize_route(self._routing[task_key], fallback_route=resolved)

        if route_overrides:
            resolved = self._normalize_route(route_overrides, fallback_route=resolved)

        return resolved

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        task_type: str = "default",
        route_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text using the routed backend/model."""
        temp = self.temperature if temperature is None else temperature
        route = self._resolve_route(task_type=task_type, route_overrides=route_overrides)
        backend: LLMBackend = route["backend"]
        model_name = route["model_name"]
        api_key = route["api_key"]
        base_url = route["base_url"]

        try:
            client = self._get_client(backend=backend, api_key=api_key, base_url=base_url)

            if backend in (LLMBackend.OPENAI, LLMBackend.OLLAMA, LLMBackend.LM_STUDIO):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                return (response.choices[0].message.content or "").strip()

            if backend == LLMBackend.GEMINI:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                model = client.GenerativeModel(model_name)
                response = model.generate_content(
                    full_prompt,
                    generation_config=client.types.GenerationConfig(
                        temperature=temp,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                return (getattr(response, "text", "") or "").strip()

            if backend == LLMBackend.ANTHROPIC:
                response = client.messages.create(
                    model=model_name,
                    max_tokens=self.max_tokens,
                    system=system_prompt or "You are a helpful assistant.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                )
                if response.content:
                    return (response.content[0].text or "").strip()
                return ""

            raise ValueError(f"Unsupported backend: {backend.value}")

        except Exception as e:
            self.last_error = f"LLM generation failed ({backend.value}/{model_name}): {str(e)}"
            raise RuntimeError(self.last_error) from e

    def extract_json(self, text: str) -> Any:
        """Extract JSON object/array from free-form LLM output."""
        if not text:
            return {}

        candidate = text.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        for block in fenced_blocks:
            block_text = block.strip()
            try:
                return json.loads(block_text)
            except json.JSONDecodeError:
                continue

        for pattern in [r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"]:
            matches = re.findall(pattern, text)
            for match in matches:
                snippet = match.strip()
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue

        return {}

    def health_check(self, task_type: str = "default", route_overrides: Optional[Dict[str, Any]] = None) -> bool:
        """Check if routed backend/model is accessible."""
        try:
            response = self.generate(
                "Respond only with OK.",
                temperature=0.0,
                task_type=task_type,
                route_overrides=route_overrides,
            )
            return "ok" in response.lower()
        except Exception:
            return False


# Global LLM instance
llm = LLMInterface()
