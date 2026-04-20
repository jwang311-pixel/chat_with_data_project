from __future__ import annotations

import requests

from app.core.config import (
    DEFAULT_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
)


class OpenRouterClient:
    def __init__(self) -> None:
        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL
        self.timeout = LLM_TIMEOUT_SECONDS
    def chat(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": OPENROUTER_APP_NAME,
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        print("OPENROUTER URL:", url)
        print("MODEL:", model)
        print("PAYLOAD:", payload)

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        print("STATUS:", response.status_code)
        print("BODY:", response.text)

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]
