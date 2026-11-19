"""Native Claude backend on the Anthropic Messages API (official `anthropic` SDK).

The harness speaks OpenAI-style messages internally so every backend shares one
agent loop. This module converts to and from the Messages API format and:

- keeps the model's raw content blocks and sends them back unchanged on the
  next turn. That includes `thinking` blocks, whose text is empty by default on
  current models but whose signature must round-trip.
- groups all tool results of one assistant turn into a single user message.
- caches prompts with an explicit breakpoint on the system prompt (tools and
  system are cached together) plus top-level automatic caching for the growing
  conversation tail.
- prices uncached input, cache writes, cache reads and output separately.

Sampling parameters are not sent unless configured, because current Claude
models reject non-default `temperature`.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .models import Budget, InflightDedup, ModelConfig, ModelResponse, ResponseCache, request_key


def to_anthropic_tools(specs: list[dict]) -> list[dict]:
    return [{"name": s["function"]["name"],
             "description": s["function"].get("description", ""),
             "input_schema": s["function"]["parameters"]} for s in specs]


def _strip_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def _is_error(content: str) -> bool:
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return False
    return isinstance(value, dict) and "error" in value


def to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Returns (system prompt, Messages API messages)."""
    system = ""
    out: list[dict] = []
    results: list[dict] = []

    def flush() -> None:
        if results:
            out.append({"role": "user", "content": list(results)})
            results.clear()

    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "system":
            if i != 0:
                raise ValueError("only a leading system message is supported")
            system = m.get("content") or ""
        elif role == "tool":
            content = m.get("content") or ""
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                     "content": content}
            if _is_error(content):
                block["is_error"] = True
            results.append(block)
        elif role == "assistant":
            flush()
            blocks = m.get("anthropic_content")
            if blocks is None:  # produced by another backend: rebuild from the generic fields
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    blocks.append({"type": "tool_use", "id": tc.get("id", ""),
                                   "name": fn.get("name", ""), "input": args})
            out.append({"role": "assistant", "content": blocks})
        elif role == "user":
            flush()
            out.append({"role": "user", "content": m.get("content") or ""})
        else:
            raise ValueError(f"unsupported message role {role!r}")
    flush()
    return system, out


def custom_ca_http_client():
    """An SDK HTTP client that verifies TLS with OpenSSL and the CA file in SSL_CERT_FILE.

    Returns None when SSL_CERT_FILE is unset, which keeps the SDK default. Use it behind a
    TLS-intercepting proxy, or in a sandbox where the macOS trust service is unreachable.
    """
    cafile = os.environ.get("SSL_CERT_FILE")
    if not cafile:
        return None
    import ssl

    import anthropic

    return anthropic.DefaultAsyncHttpxClient(verify=ssl.create_default_context(cafile=cafile))


class AnthropicClient:
    def __init__(self, cfg: ModelConfig, cache: ResponseCache | None, budget: Budget):
        import anthropic

        key = os.environ.get(cfg.api_key_env or "ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(f"model {cfg.name!r}: set {cfg.api_key_env} in your shell or in a "
                               f".env file in the project folder")
        kwargs: dict[str, Any] = {"api_key": key, "max_retries": 4, "timeout": 300.0}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if (http_client := custom_ca_http_client()) is not None:
            kwargs["http_client"] = http_client
        self.cfg, self.cache, self.budget, self.name = cfg, cache, budget, cfg.name
        self._dedup = InflightDedup()
        self.client = anthropic.AsyncAnthropic(**kwargs)

    def cost(self, usage: dict) -> float:
        c = self.cfg
        write = c.price_cache_write_per_m
        read = c.price_cache_read_per_m
        write = 1.25 * c.price_in_per_m if write is None else write
        read = 0.1 * c.price_in_per_m if read is None else read
        return ((usage.get("input_tokens") or 0) * c.price_in_per_m
                + (usage.get("cache_creation_input_tokens") or 0) * write
                + (usage.get("cache_read_input_tokens") or 0) * read
                + (usage.get("output_tokens") or 0) * c.price_out_per_m) / 1e6

    def build_params(self, messages: list[dict], tools: list[dict]) -> dict:
        system, msgs = to_anthropic_messages(messages)
        params: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "messages": msgs,
            "tools": to_anthropic_tools(tools),
            "cache_control": {"type": "ephemeral"},  # automatic breakpoint on the conversation tail
            **self.cfg.extra_params,
        }
        if system:
            params["system"] = [{"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}}]
        if self.cfg.temperature is not None:
            params["temperature"] = self.cfg.temperature
        return params

    async def complete(self, messages: list[dict], tools: list[dict], *,
                       attempt: int = 0) -> ModelResponse:
        params = self.build_params(messages, tools)
        key = request_key(params, attempt)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return ModelResponse(hit["message"], hit["input_tokens"], hit["output_tokens"], True,
                                 hit.get("cache_read_tokens", 0), hit.get("cache_write_tokens", 0),
                                 hit.get("cost_usd", 0.0))
        record, shared = await self._dedup.run(key, lambda: self._fetch(params, key))
        return ModelResponse(record["message"], record["input_tokens"], record["output_tokens"],
                             shared, record["cache_read_tokens"], record["cache_write_tokens"],
                             record["cost_usd"])

    async def _fetch(self, params: dict, key: str) -> dict:
        self.budget.check()
        resp = await self.client.messages.create(**params)
        data = resp.to_dict()
        blocks = [_strip_none(b) for b in data.get("content") or []]
        stop = data.get("stop_reason")
        text = "\n".join(b["text"] for b in blocks if b.get("type") == "text" and b.get("text"))
        message: dict[str, Any] = {"role": "assistant", "content": text,
                                   "anthropic_content": blocks, "stop_reason": stop}
        if stop != "max_tokens":  # a turn cut off by max_tokens may hold an incomplete call
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))}}
                     for b in blocks if b.get("type") == "tool_use"]
            if calls:
                message["tool_calls"] = calls
        if stop == "refusal":
            message["refusal"] = _strip_none(data.get("stop_details") or {}) or {"type": "refusal"}
        usage = data.get("usage") or {}
        cost = self.cost(usage)
        self.budget.charge(cost)
        record = {
            "message": message,
            "input_tokens": usage.get("input_tokens") or 0,
            "output_tokens": usage.get("output_tokens") or 0,
            "cache_read_tokens": usage.get("cache_read_input_tokens") or 0,
            "cache_write_tokens": usage.get("cache_creation_input_tokens") or 0,
            "cost_usd": cost,
        }
        if self.cache is not None:
            self.cache.put(key, record)
        return record
