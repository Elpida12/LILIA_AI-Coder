"""OpenAI-compatible LLM client with tool calling support + async + caching."""

import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

import httpx


class LLMError(Exception):
    pass


def _hash_request(messages: list[dict], tools: list[dict] | None, temperature: float,
                  model: str = "", max_tokens: int = 0, thinking_budget: int = 0) -> str:
    payload = {"m": messages, "t": tools, "temp": temperature,
               "model": model, "max_tok": max_tokens, "think": thinking_budget}
    data = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:24]


class LLMBackend:
    def __init__(self, base_url: str, model: str, api_key: str, context_size: int,
                 reasoning_format: str, temperature: float, max_tokens: int, logger,
                 cache_dir: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.context_size = context_size
        self.reasoning_format = reasoning_format
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger

        headers = {}
        if api_key and api_key != "no-key-needed":
            headers["Authorization"] = f"Bearer {api_key}"

        self.client = httpx.AsyncClient(
            timeout=None, headers=headers,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        self._memory_cache: dict[str, dict] = {}
        if cache_dir:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = None

        # Circuit breaker state (Fix #5C)
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    async def achat(self, messages: list[dict], tools: list[dict] | None = None,
                    agent_name: str = "unknown", thinking_budget: int = 2048) -> dict:
        cache_key = _hash_request(messages, tools, self.temperature,
                                   model=self.model, max_tokens=self.max_tokens,
                                   thinking_budget=thinking_budget)
        cached = self._memory_cache.get(cache_key)
        if cached:
            self.logger.debug(f"LLM cache hit for {agent_name}")
            return cached
        if self._cache_dir:
            cache_file = self._cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                self._memory_cache[cache_key] = cached
                self.logger.debug(f"LLM disk cache hit for {agent_name}")
                return cached

        # Circuit breaker check (Fix #5C)
        if self._consecutive_failures >= 5 and time.time() < self._circuit_open_until:
            raise LLMError("Circuit breaker open — LLM API appears down")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if self.reasoning_format == "deepseek":
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        start = time.time()
        error = None
        for attempt in range(3):
            try:
                resp = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                break
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as exc:
                # Fix #5A: Add jitter to backoff delays
                base = 2 ** (attempt + 1)
                jitter = random.uniform(0.5, 1.5)  # ±50% jitter
                wait = base * jitter
                if attempt < 2:
                    self.logger.warning(
                        f"LLM failed ({exc.__class__.__name__}), attempt {attempt+2}/3 in {wait:.1f}s",
                        agent=agent_name,
                    )
                    await asyncio.sleep(wait)
                else:
                    error = exc
        else:
            # All 3 retries failed (Fix #5C: update circuit breaker)
            self._consecutive_failures += 1
            if self._consecutive_failures >= 5:
                self._circuit_open_until = time.time() + 60  # 60s cooldown
            raise LLMError(f"LLM at {self.base_url} unreachable after 3 attempts: {error}")

        duration = time.time() - start
        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        thinking = choice.get("reasoning_content") or ""
        tool_calls_raw = choice.get("tool_calls") or []
        usage = data.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        comp_tok = usage.get("completion_tokens", 0)

        # Fix #1A: Extract finish_reason from API response
        finish_reason = data["choices"][0].get("finish_reason", "stop")

        self.logger.llm_call(
            agent=agent_name, messages=messages,
            response=content[:500], thinking=thinking[:500],
            tokens_prompt=prompt_tok, tokens_completion=comp_tok,
            duration=duration,
            finish_reason=finish_reason,  # Fix #1D: Pass finish_reason to logger
        )

        # Reset circuit breaker on success (Fix #5C)
        self._consecutive_failures = 0

        result = {
            "content": content,
            "thinking": thinking,
            "tool_calls": self._normalize_tool_calls(tool_calls_raw),
            "tokens_prompt": prompt_tok,
            "tokens_completion": comp_tok,
            "finish_reason": finish_reason,  # Fix #1A
            "truncated": finish_reason == "length",  # Fix #1A
        }

        self._memory_cache[cache_key] = result
        if self._cache_dir:
            (self._cache_dir / f"{cache_key}.json").write_text(
                json.dumps(result, default=str), encoding="utf-8"
            )
        return result

    def _normalize_tool_calls(self, raw: list[dict]) -> list[dict]:
        out = []
        for tc in raw:
            fn = tc.get("function", {})
            name = (fn.get("name") or "").strip()
            args_raw = fn.get("arguments", "{}")
            malformed = False
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError as e:
                    # Fix #1B: Log warning instead of silently falling back
                    self.logger.warning(
                        f"Malformed tool call arguments for '{name}': {e}. "
                        f"Raw preview: {args_raw[:200]}"
                    )
                    args = {}
                    malformed = True
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            out.append({
                "id": tc.get("id") or f"tc_{len(out)}",
                "name": name,
                "arguments": args,
                "_malformed": malformed,  # Fix #1B: Flag for agent loop
            })
        return out

    async def close(self):
        await self.client.aclose()