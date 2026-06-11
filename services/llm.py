"""Thin async client for an OpenAI-compatible chat endpoint.

Defaults to LM Studio's local server (config.LLM_ENDPOINT) — free, local, and
private, so edgy persona content never leaves the machine. Any OpenAI-style
`/v1/chat/completions` server works (LM Studio, llama.cpp server, Ollama's
OpenAI shim, or a hosted API). Returns None on any failure so callers degrade
gracefully (the bot stays up if the model server is off).
"""

import logging
import asyncio

import aiohttp

import config

_last_error = None
_chat_lock = asyncio.Lock()
_resolved_model = None


def last_error() -> str | None:
    return _last_error


async def _pick_model(session) -> str:
    """The model id to send. Newer LM Studio rejects the legacy 'local' id, so
    if config.LLM_MODEL is blank/'local' we auto-detect the first non-embedding
    loaded model. A real configured id is used as-is (also how the A/B picks
    each model)."""
    global _resolved_model
    cfg = (config.LLM_MODEL or "").strip()
    if cfg and cfg.lower() != "local":
        return cfg
    if _resolved_model:
        return _resolved_model
    try:
        base = config.LLM_ENDPOINT.split("/v1/")[0]
        async with session.get(base + "/v1/models") as r:
            ids = [m["id"] for m in (await r.json()).get("data", [])]
        for mid in ids:
            if "embed" not in mid.lower():
                _resolved_model = mid
                return mid
    except Exception:
        pass
    return "local"


async def chat(messages, max_tokens: int = 120, temperature: float = 0.85,
               stop=None) -> str | None:
    global _last_error
    _last_error = None
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if stop:
        payload["stop"] = stop
    try:
        async with _chat_lock:
            timeout = aiohttp.ClientTimeout(total=config.LLM_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload["model"] = await _pick_model(session)
                async with session.post(config.LLM_ENDPOINT, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        _last_error = f"HTTP {resp.status}: {body[:200]}"
                        logging.warning(f"LLM endpoint {_last_error}")
                        return None
                    data = await resp.json()
                    return (data["choices"][0]["message"]["content"] or "").strip()
    except asyncio.TimeoutError:
        _last_error = f"timeout after {config.LLM_TIMEOUT}s"
        logging.warning(f"LLM call failed ({config.LLM_ENDPOINT}): {_last_error}")
        return None
    except Exception as e:
        _last_error = f"{type(e).__name__}: {e!r}"
        logging.warning(f"LLM call failed ({config.LLM_ENDPOINT}): {_last_error}")
        return None


async def available() -> bool:
    """Quick reachability check for local/remote OpenAI-compatible endpoints."""
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    try:
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(base + "/v1/models") as resp:
                return resp.status == 200
    except Exception:
        return False
