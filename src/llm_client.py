from __future__ import annotations

"""
Ollama LLM Client
==================
Wrapper around the Ollama REST API for local model inference.
Handles connection checking, generation, speed benchmarking,
and checkpoint-based resume on failure.
"""

import time
import requests


class OllamaClient:
    """Client for the locally-running Ollama inference server."""

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the Ollama client.

        Args:
            model_name: Name of the model to use (must already be pulled in Ollama).
            base_url:   Ollama server URL (default: http://localhost:11434).
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        """
        Check if the Ollama server is running and the specified model is loaded.

        Returns:
            True if server is reachable and model is available.
        """
        try:
            # Check server is up
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()

            # Check our model is in the list
            models = [m["name"] for m in resp.json().get("models", [])]
            # Ollama model names can be "qwen2.5:7b" or "qwen2.5:7b-instruct-..."
            # Match by prefix: "qwen2.5:7b" matches "qwen2.5:7b" and "qwen2.5:7b-q4_0"
            return any(
                m == self.model_name or m.startswith(self.model_name)
                for m in models
            )
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            return False

    def list_models(self) -> list[str]:
        """
        List all models currently available in the local Ollama installation.

        Returns:
            List of model name strings (e.g., ["qwen2.5:7b", "mistral:7b"]).
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            return []

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        """
        Send a prompt to Ollama and return the generated text.

        Uses the /api/generate endpoint with streaming disabled for simplicity.

        Args:
            prompt:        The user/instruction prompt.
            system_prompt: System-level instructions (role, constraints).
            temperature:   Sampling temperature (0.0-1.0). Lower = more deterministic.
            max_tokens:    Maximum tokens to generate in the response.

        Returns:
            The generated text string.

        Raises:
            ConnectionError: If Ollama server is unreachable.
            RuntimeError:    If the model returns an error.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300,  # 5 min timeout for long generations
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code} — {e.response.text}")

        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"Ollama model error: {data['error']}")

        return data.get("response", "")

    def estimate_speed(self, sample_text: str = None) -> dict:
        """
        Run a small benchmark to estimate tokens/second on the current hardware.

        Sends a short prompt and measures the time to generate ~200 tokens.
        Uses this to estimate total processing time for a book.

        Args:
            sample_text: Optional custom text for the benchmark prompt.

        Returns:
            Dict with keys:
              - "tokens_per_second"       : float
              - "estimated_minutes_200pg" : float  (estimate for a 200-page book)
              - "model_name"              : str
        """
        if sample_text is None:
            sample_text = (
                "Explain the concept of entropy in simple terms. "
                "Write about 200 words."
            )

        start = time.time()
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": sample_text,
                "stream": False,
                "options": {"num_predict": 200},
            },
            timeout=120,
        )
        elapsed = time.time() - start
        data = resp.json()

        # Ollama returns eval_count (tokens generated) and eval_duration (nanoseconds)
        eval_count = data.get("eval_count", 200)
        eval_duration_ns = data.get("eval_duration", elapsed * 1e9)
        tokens_per_second = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0

        # Estimate for a 200-page book:
        # ~70,000 words → ~93,000 tokens input → ~35 chunks of 3000 tokens
        # Each chunk produces ~3000 output tokens
        # Total output tokens ≈ 35 * 3000 = 105,000
        total_output_tokens = 105_000
        estimated_seconds = total_output_tokens / tokens_per_second if tokens_per_second > 0 else 0
        estimated_minutes = estimated_seconds / 60

        return {
            "tokens_per_second": round(tokens_per_second, 1),
            "estimated_minutes_200pg": round(estimated_minutes, 1),
            "model_name": self.model_name,
        }
