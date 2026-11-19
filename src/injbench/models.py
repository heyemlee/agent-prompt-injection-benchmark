"""Model clients: any OpenAI-compatible endpoint, plus scripted reference agents.

Responses from real models are cached on disk keyed by the full request, so
re-running an analysis never pays twice. A shared budget stops new API calls
once the configured dollar cap is reached.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class ModelConfig:
    name: str
    kind: str = "openai_compat"  # openai_compat | anthropic | scripted
    model: str = ""
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = None  # None: not sent (some models reject it)
    max_tokens: int = 16000  # anthropic only; thinking counts toward it
    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0
    price_cache_write_per_m: float | None = None  # default 1.25x input (5-minute TTL)
    price_cache_read_per_m: float | None = None  # default 0.1x input
    extra_params: dict = field(default_factory=dict)
    behavior: str = "gold"  # scripted only: gold | gullible | null | refuser | leaker
    # SDK retries for 429 / 5xx / connection errors, with exponential backoff (and `retry-after`
    # when the provider sends it). None: the client's default (3 OpenAI-compatible, 4 Anthropic).
    # Raise it for providers with tight per-minute quotas: about 12 retries span a minute window.
    max_retries: int | None = None
    # A model served on your own machine (llama.cpp, vLLM, LM Studio). It costs nothing, so it
    # needs no price and the budget cap ignores it, and it needs no API key unless its server was
    # started with one (then name the variable in api_key_env).
    local: bool = False
    # Read the server address from this environment variable (or .env) instead of base_url, so a
    # machine's network name stays out of configs that are committed to a public repository.
    base_url_env: str | None = None
    # Seconds before a request is abandoned and retried. A laptop serving several episodes at
    # once can need minutes for one turn, so local models need far more than the default.
    timeout_s: float = 120.0


@dataclass
class ModelResponse:
    message: dict
    input_tokens: int = 0  # uncached input tokens
    output_tokens: int = 0
    cached: bool = False  # served from the local response cache (no API call)
    cache_read_tokens: int = 0  # provider-side prompt cache
    cache_write_tokens: int = 0
    cost_usd: float = 0.0


class ModelClient(Protocol):
    name: str

    async def complete(self, messages: list[dict], tools: list[dict], *,
                       attempt: int = 0) -> ModelResponse: ...


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, limit_usd: float | None):
        self.limit = limit_usd
        self.spent = 0.0

    def check(self) -> None:
        if self.limit is not None and self.spent >= self.limit:
            raise BudgetExceeded(f"budget of ${self.limit:.2f} reached (spent ${self.spent:.4f})")

    def charge(self, usd: float) -> None:
        self.spent += usd


class ResponseCache:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self.conn.commit()

    def get(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
        self.conn.commit()


class InflightDedup:
    """Coalesces identical concurrent requests into one API call.

    Defense arms share prompts until a tool result differs, so identical requests from different
    arms often run concurrently. Without this, each fires its own call, gets a different sample,
    and the cache keeps only one, so arms stop sharing responses and a re-run is not reproducible.
    With it, an identical request is sampled once and the arms differ only after the defense
    changes what the model sees.
    """

    def __init__(self):
        self._inflight: dict[str, asyncio.Future] = {}

    async def run(self, key: str, fetch) -> tuple[dict, bool]:
        """Returns (result, shared). `shared` is True when another caller made the API call."""
        if key in self._inflight:
            return await asyncio.shield(self._inflight[key]), True
        fut = asyncio.get_running_loop().create_future()
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # mark errors retrieved
        self._inflight[key] = fut
        try:
            result = await fetch()
        except BaseException as e:
            fut.set_exception(e)
            raise
        else:
            fut.set_result(result)
            return result, False
        finally:
            del self._inflight[key]


OPENAI_MESSAGE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")


def sanitize_messages(messages: list[dict]) -> list[dict]:
    """Only standard chat-completions fields are sent. Bookkeeping such as the stop reason,
    reasoning text or serving provider stays in the transcript only."""
    return [{k: m[k] for k in OPENAI_MESSAGE_KEYS if k in m} for m in messages]


def request_key(params: dict, attempt: int) -> str:
    blob = json.dumps({"params": params, "attempt": attempt}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class OpenAICompatClient:
    """Chat-completions client for OpenAI, OpenRouter, Together, Tinker's sampler, local servers."""

    def __init__(self, cfg: ModelConfig, cache: ResponseCache | None, budget: Budget):
        from openai import AsyncOpenAI  # imported lazily so offline tests need no network setup

        key = os.environ.get(cfg.api_key_env)
        if not key and cfg.local:
            key = "local"  # a server started without --api-key accepts any key
        if not key:
            raise RuntimeError(f"model {cfg.name!r}: set {cfg.api_key_env} in your shell or in a "
                               f".env file in the project folder")
        base_url = cfg.base_url
        if cfg.base_url_env:
            base_url = os.environ.get(cfg.base_url_env) or base_url
            if not base_url:
                raise RuntimeError(f"model {cfg.name!r}: set {cfg.base_url_env} to the server's "
                                   f"address (like http://host:8000/v1) in your shell or in .env")
        self.cfg, self.cache, self.budget = cfg, cache, budget
        self.name = cfg.name
        self._dedup = InflightDedup()
        kwargs: dict = {"base_url": base_url, "api_key": key, "timeout": cfg.timeout_s,
                        "max_retries": 3 if cfg.max_retries is None else cfg.max_retries}
        if (cafile := os.environ.get("SSL_CERT_FILE")):
            import ssl

            import openai

            kwargs["http_client"] = openai.DefaultAsyncHttpxClient(
                verify=ssl.create_default_context(cafile=cafile))
        self.client = AsyncOpenAI(**kwargs)

    async def complete(self, messages, tools, *, attempt: int = 0) -> ModelResponse:
        params = {"model": self.cfg.model, "messages": sanitize_messages(messages), "tools": tools,
                  **self.cfg.extra_params}
        if self.cfg.temperature is not None:
            params["temperature"] = self.cfg.temperature
        key = request_key(params, attempt)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return ModelResponse(hit["message"], hit["input_tokens"], hit["output_tokens"], True,
                                 cost_usd=hit.get("cost_usd", 0.0))
        record, shared = await self._dedup.run(key, lambda: self._fetch(params, key))
        return ModelResponse(record["message"], record["input_tokens"], record["output_tokens"],
                             shared, cost_usd=record["cost_usd"])

    async def _fetch(self, params: dict, key: str) -> dict:
        self.budget.check()
        resp = await self.client.chat.completions.create(**params)
        first = resp.choices[0]
        choice = first.message
        finish = getattr(first, "finish_reason", None)
        message: dict = {"role": "assistant", "content": choice.content, "stop_reason": finish}
        calls = []
        for tc in choice.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            calls.append({"id": tc.id, "type": "function",
                          "function": {"name": fn.name, "arguments": fn.arguments}})
        if calls and finish != "length":  # a turn cut off by the token limit may hold a broken call
            message["tool_calls"] = calls
        if finish == "content_filter":
            message["refusal"] = {"type": "content_filter"}
        extra = getattr(choice, "model_extra", None) or {}
        reasoning = extra.get("reasoning") or extra.get("reasoning_content")  # OpenRouter / SiliconFlow
        if reasoning:
            message["reasoning"] = reasoning
        provider = (getattr(resp, "model_extra", None) or {}).get("provider")
        if provider:
            message["provider"] = provider
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        cost = tin / 1e6 * self.cfg.price_in_per_m + tout / 1e6 * self.cfg.price_out_per_m
        self.budget.charge(cost)
        record = {"message": message, "input_tokens": tin, "output_tokens": tout, "cost_usd": cost}
        if self.cache is not None:
            self.cache.put(key, record)
        return record


class ScriptedClient:
    """Replays a fixed action list, one tool call per turn, then a final reply.

    Used for reference agents (gold, gullible, null, refuser, leaker) that
    validate the graders without any API calls.
    """

    def __init__(self, name: str, actions: list[tuple[str, dict]], final_reply: str):
        self.name = name
        self.actions = list(actions)
        self.final_reply = final_reply

    async def complete(self, messages, tools, *, attempt: int = 0) -> ModelResponse:
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if n < len(self.actions):
            tool, args = self.actions[n]
            call = {"id": f"call_{n}", "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(args)}}
            return ModelResponse({"role": "assistant", "content": None, "tool_calls": [call]})
        return ModelResponse({"role": "assistant", "content": self.final_reply})
