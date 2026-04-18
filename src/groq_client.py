from __future__ import annotations

"""
Groq LLM Client
================
Wrapper around the Groq API (OpenAI-compatible) for cloud-based inference.
Drop-in replacement for OllamaClient with the same generate() interface.

Uses plain `requests` — no extra SDK needed. Groq's API is OpenAI-compatible.

Requires:
    export GROQ_API_KEY='your-api-key'
    or set GROQ_API_KEY in config.py
    Get a free key at: https://console.groq.com/
"""

import os
import time
import requests


class GroqClient:
    """Client for the Groq API (OpenAI-compatible)."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: str = None,
    ):
        """
        Initialize the Groq client.

        Args:
            model_name: Groq model name (default: llama-3.3-70b-versatile).
            api_key:    API key. If None, reads from GROQ_API_KEY env var or config.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")

        # Fall back to config.py
        if not self.api_key:
            try:
                from config import GROQ_API_KEY
                self.api_key = GROQ_API_KEY
            except (ImportError, AttributeError):
                pass

        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set it with:\n"
                "  export GROQ_API_KEY='your-api-key'\n"
                "Get a free key at: https://console.groq.com/"
            )

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def is_available(self) -> bool:
        """
        Check if the Groq API is reachable and the API key is valid.
        Uses the models endpoint to avoid consuming generation quota.
        """
        try:
            resp = requests.get(
                f"{self.BASE_URL}/models",
                headers=self._headers,
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        """
        Send a prompt to Groq and return the generated text.

        Mirrors the OllamaClient.generate() interface for drop-in compatibility.

        Args:
            prompt:        The user/instruction prompt.
            system_prompt: System-level instructions (role, constraints).
            temperature:   Sampling temperature (0.0-1.0).
            max_tokens:    Maximum tokens to generate.

        Returns:
            The generated text string.

        Raises:
            RuntimeError: If the API returns an error after retries.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Retry with backoff for rate limits
        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._headers,
                    json=payload,
                    timeout=300,
                )

                if resp.status_code in (429, 413):  # 429=rate limit, 413=TPM exceeded
                    if attempt < max_retries:
                        # Use Retry-After header; for TPM issues, wait 10s minimum
                        wait = float(resp.headers.get("retry-after", max(10, 2 * (attempt + 1))))
                        if wait > 60:
                            # Daily quota exhausted — don't wait hours
                            raise RuntimeError(
                                f"Groq daily quota exhausted (retry-after: {wait:.0f}s). "
                                "Try again later or use a different model."
                            )
                        print(f"  ⏳ Rate limited, retrying in {wait:.0f}s... ({attempt+1}/{max_retries})")
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"Groq rate limit exceeded after {max_retries} retries")

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except requests.HTTPError as e:
                raise RuntimeError(f"Groq API error: {e.response.status_code} — {e.response.text}")
            except (requests.ConnectionError, requests.Timeout) as e:
                raise RuntimeError(f"Groq connection error: {e}")

    def estimate_speed(self, sample_text: str = None) -> dict:
        """
        Return estimated speed for Groq API.
        Groq is extremely fast (LPU inference) — no benchmark needed.
        """
        return {
            "tokens_per_second": 300.0,   # Groq is blazing fast
            "estimated_minutes_200pg": 6.0,
            "model_name": self.model_name,
        }
