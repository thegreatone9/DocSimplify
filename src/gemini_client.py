from __future__ import annotations

"""
Gemini LLM Client
==================
Wrapper around the Google Gemini API for cloud-based inference.
Drop-in replacement for OllamaClient with the same generate() interface.

Requires:
    pip install google-genai
    export GEMINI_API_KEY='your-api-key'
"""

import os
import time


class GeminiClient:
    """Client for the Google Gemini API."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str = None,
    ):
        """
        Initialize the Gemini client.

        Args:
            model_name: Gemini model name (default: gemini-2.5-flash).
            api_key:    API key. If None, reads from GEMINI_API_KEY env var.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

        # Fall back to config.py
        if not self.api_key:
            try:
                from config import GEMINI_API_KEY
                self.api_key = GEMINI_API_KEY
            except (ImportError, AttributeError):
                pass

        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set it with:\n"
                "  export GEMINI_API_KEY='your-api-key'\n"
                "Get a free key at: https://aistudio.google.com/"
            )

        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def is_available(self) -> bool:
        """
        Check if the Gemini API is reachable and the API key is valid.
        Uses models.get() to avoid consuming generation quota.

        Returns:
            True if the API responds successfully.
        """
        try:
            self._client.models.get(model=self.model_name)
            return True
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 65536,
    ) -> str:
        """
        Send a prompt to Gemini and return the generated text.

        Mirrors the OllamaClient.generate() interface for drop-in compatibility.
        Disables thinking mode for direct, fast output.

        Args:
            prompt:        The user/instruction prompt.
            system_prompt: System-level instructions (role, constraints).
            temperature:   Sampling temperature (0.0-1.0).
            max_tokens:    Maximum output tokens.

        Returns:
            The generated text string.

        Raises:
            RuntimeError: If the API returns an error after retries.
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        # Disable thinking mode for 2.5 models (thinking tokens eat output limit)
        if "2.5" in self.model_name:
            config.thinking_config = types.ThinkingConfig(thinking_budget=0)
        if system_prompt:
            config.system_instruction = system_prompt

        # Retry with backoff for rate limits
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries:
                        wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                        print(f"  ⏳ Rate limited, retrying in {wait}s... ({attempt+1}/{max_retries})")
                        time.sleep(wait)
                        continue
                raise RuntimeError(f"Gemini API error: {e}")

    def estimate_speed(self, sample_text: str = None) -> dict:
        """
        Return estimated speed for Gemini API.
        Unlike Ollama, no benchmark needed — Gemini speed is server-side
        and consistent. Avoids wasting quota on a throwaway call.
        """
        return {
            "tokens_per_second": 200.0,   # Gemini is very fast
            "estimated_minutes_200pg": 9.0,
            "model_name": self.model_name,
        }
