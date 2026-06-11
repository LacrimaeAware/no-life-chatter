"""Thin async client for an OpenAI-compatible chat endpoint.

Defaults to LM Studio's local server (config.LLM_ENDPOINT) — free, local, and
private, so edgy persona content never leaves the machine. Any OpenAI-style
`/v1/chat/completions` server works (LM Studio, llama.cpp server, Ollama's
OpenAI shim, or a hosted API). Returns None on any failure so callers degrade
gracefully (the bot stays up if the model server is off).
"""

import logging

import aiohttp

import config


async def chat(messages, max_tokens: int = 120, temperature: float = 0.85,
               stop=None) -> str | None:
    payload = {
        "model": config.LLM_MODEL or "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if stop:
        payload["stop"] = stop
    try:
        timeout = aiohttp.ClientTimeout(total=config.LLM_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(config.LLM_ENDPOINT, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logging.warning(f"LLM endpoint {resp.status}: {body[:200]}")
                    return None
                data = await resp.json()
                return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        logging.warning(f"LLM call failed ({config.LLM_ENDPOINT}): {e}")
        return None


async def available() -> bool:
    """Quick reachability check (used to decide LLM vs Markov fallback)."""
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    try:
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(base + "/v1/models") as resp:
                return resp.status == 200
    except Exception:
        return False
