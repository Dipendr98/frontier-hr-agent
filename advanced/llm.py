"""
Provider-agnostic LLM client.

SCOPE — READ THIS BEFORE ASSUMING WHAT THE LLM DOES HERE.

This module serves two callers with deliberately different privileges:

  Mode 1 (advanced/orchestration/workflow.py) — the LLM is a BOUNDED LANGUAGE
  LAYER. It phrases an explanation AFTER the structured decision already
  exists. Its output is never read back into control flow, so the pipeline
  produces identical decisions with or without a provider.

  Mode 2/3 (advanced/orchestration/llm_agent.py) — the LLM genuinely drives:
  it picks tools, arguments and stopping point. In Mode 3 every claim it makes
  is then re-derived from the tool layer and checked by the same deterministic
  Critic, so model authority is bounded by verification rather than by scope.

Judges will NOT have your API key, so Mode 1 — which every headline number is
computed on — runs with NO key at all. Modes 2 and 3 require one and say so.

Configure via environment variables (see .env.example):

  LLM_PRESET     nvidia | gemini | groq | openrouter | openai | anthropic | ollama
  LLM_API_KEY    your key for that provider
  LLM_MODEL      optional override of the preset's default model

Or fully manual:

  LLM_PROVIDER   anthropic | openai_compatible
  LLM_BASE_URL   e.g. https://integrate.api.nvidia.com/v1
  LLM_API_KEY    your key
  LLM_MODEL      e.g. nvidia/nemotron-3-super-120b-a12b

Anything speaking the OpenAI chat-completions format works through
`openai_compatible` — NVIDIA NIM, Gemini (OpenAI-compat endpoint), Groq,
OpenRouter, Together, local vLLM/Ollama. Only base_url + key + model change.

Run `python -m advanced.doctor` to check a configured provider end to end,
including whether it actually supports the tool calling Modes 2/3 need.
"""
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- config

# Default models are chosen for TOOL-CALLING RELIABILITY AND LATENCY, not raw
# benchmark score. Mode 2 issues up to ~8 sequential round trips per case, so a
# model that takes 60s per turn makes a single case take ten minutes. We
# measured this: the previous NVIDIA default (deepseek-v4-pro) completed one
# EMP1180 investigation in 575 seconds. nemotron-3-super returns a correct
# tool call in well under a second. Both are "good" models; only one is usable
# inside an agent loop.
PRESETS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "nvidia/nemotron-3-super-120b-a12b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "anthropic": {
        "base_url": "",                       # native API, not OpenAI-compatible
        # Current model. A previous fix replaced a non-existent ID with
        # claude-3-5-sonnet-20241022, which exists but is two generations old;
        # override with LLM_MODEL if you want a cheaper tier.
        "model": "claude-opus-5",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1",
    },
}

# Indicative USD per 1M tokens, used only to report an order-of-magnitude cost
# per case. Marked ESTIMATED everywhere it surfaces — provider pricing changes
# and we are not going to pretend this is a billing record.
PRICING_PER_MTOK = {
    "gpt-4o-mini":                      {"in": 0.15,  "out": 0.60},
    "claude-opus-5":                    {"in": 5.00,  "out": 25.00},
    "claude-sonnet-5":                  {"in": 3.00,  "out": 15.00},
    "claude-haiku-4-5":                 {"in": 1.00,  "out": 5.00},
    "claude-sonnet-4-6":                {"in": 3.00,  "out": 15.00},
    "gemini-2.0-flash":                 {"in": 0.10,  "out": 0.40},
    "llama-3.3-70b-versatile":          {"in": 0.59,  "out": 0.79},
    "meta-llama/llama-3.3-70b-instruct": {"in": 0.12, "out": 0.30},
    "nvidia/nemotron-3-super-120b-a12b": {"in": 0.00, "out": 0.00},  # free tier
}
DEFAULT_PRICING = {"in": 0.50, "out": 1.50}

PROVIDER = os.environ.get("LLM_PROVIDER", "none").strip().lower()
BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "")
TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "90"))
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "6"))

# Client-side throttle.
#
# Free and trial tiers rate-limit hard, and an agent loop is exactly the shape
# of traffic that trips them: many short sequential calls, multiplied by however
# many cases run in parallel. We learned this the direct way — a 6-case Mode 2/3
# comparison at 3 workers got HTTP 429 on nine of twelve runs, and because a
# failed run still produced a row, the comparison table quietly reported those
# as zero-quality cases rather than as infrastructure failures. A wrong number
# is worse than a missing one.
#
# So: self-limit instead of discovering the limit by hitting it. A semaphore
# caps concurrency and a minimum interval spaces out request starts across all
# threads. Both are env-tunable, because the right values are a property of your
# account, not of this code.
MAX_CONCURRENCY = int(os.environ.get("LLM_MAX_CONCURRENCY", "2"))
MIN_INTERVAL = float(os.environ.get("LLM_MIN_INTERVAL", "0.35"))

_slots = threading.Semaphore(max(1, MAX_CONCURRENCY))
_pace_lock = threading.Lock()
_last_start = [0.0]


class _Throttle:
    """Cap concurrent requests and space out their start times."""

    def __enter__(self):
        _slots.acquire()
        if MIN_INTERVAL > 0:
            with _pace_lock:
                wait = _last_start[0] + MIN_INTERVAL - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                _last_start[0] = time.monotonic()
        return self

    def __exit__(self, *exc):
        _slots.release()
        return False

# Provider-specific request fields that are not part of the OpenAI schema.
#
# Several NVIDIA NIM models are reasoning models: they emit a chain of thought
# into a separate `reasoning_content` field, and when the response is truncated
# that reasoning can spill into `content` instead. We do not want a paragraph of
# deliberation reaching an HR reviewer, and the thinking tokens are pure latency
# for a task where the reasoning is already externalised into tools. Turning it
# off is a per-provider flag, so it is applied per-preset rather than sent to
# every provider — OpenAI rejects unknown top-level fields outright.
PRESET_EXTRA_BODY = {
    "nvidia": {"chat_template_kwargs": {"thinking": False}},
}
EXTRA_BODY = {}

# Reasoning models can burn the whole budget before emitting a single visible
# token, so no request is allowed below this ceiling.
MIN_MAX_TOKENS = 512

# Mode 1's optional prose layer ("narration").
#
# It costs one completion per agent per case and, by design, changes no
# decision — so on a 342-employee cohort it buys nicer wording for ~700 API
# calls and several minutes. Batch entry points therefore switch it OFF and
# single-case runs leave it on; `LLM_NARRATE=0/1` overrides either way.
# Making this explicit rather than incidental matters: a reader should be able
# to tell from the command whether a run cost anything.
NARRATION = os.environ.get("LLM_NARRATE", "1").strip().lower() not in ("0", "false", "no")


def set_narration(enabled: bool):
    """Enable/disable Mode 1's optional prose layer. Never affects decisions."""
    global NARRATION
    NARRATION = bool(enabled)

_preset = os.environ.get("LLM_PRESET", "").strip().lower()
if _preset in PRESETS:
    if PROVIDER == "none":
        PROVIDER = "anthropic" if _preset == "anthropic" else "openai_compatible"
    BASE_URL = BASE_URL or PRESETS[_preset]["base_url"].rstrip("/")
    MODEL = MODEL or PRESETS[_preset]["model"]
    EXTRA_BODY = dict(PRESET_EXTRA_BODY.get(_preset, {}))

# Escape hatch for any provider field we have not anticipated.
_extra = os.environ.get("LLM_EXTRA_BODY", "").strip()
if _extra:
    try:
        EXTRA_BODY.update(json.loads(_extra))
    except json.JSONDecodeError:
        raise SystemExit("LLM_EXTRA_BODY is not valid JSON")


def is_enabled() -> bool:
    """True when a real LLM is configured. Everything in Mode 1 degrades gracefully if False."""
    if PROVIDER == "anthropic":
        return bool(API_KEY)
    if PROVIDER == "openai_compatible":
        return bool(BASE_URL and MODEL)  # local servers may need no key
    return False


def active_model() -> str:
    return MODEL or (PRESETS["anthropic"]["model"] if PROVIDER == "anthropic" else "")


def describe() -> str:
    """Recorded in trajectories so every result is attributable to a provider."""
    if not is_enabled():
        return "provider=none (deterministic rule-based fallback)"
    if PROVIDER == "anthropic":
        return f"provider=anthropic model={active_model()}"
    return f"provider=openai_compatible base_url={BASE_URL} model={MODEL}"


# ---------------------------------------------------------------- accounting


class UsageMeter:
    """
    Tracks tokens, wall clock, retries and an ESTIMATED cost across calls.

    The hackathon evaluation format asks for a cost per task. Without this the
    row would be a guess; with it, it is at least a measured token count times
    a published rate. Thread-safe because the cohort runner fans out cases.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.calls = 0
            self.retries = 0
            self.failures = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.seconds = 0.0

    def record(self, usage: dict, seconds: float, retries: int = 0):
        with self._lock:
            self.calls += 1
            self.retries += retries
            self.seconds += seconds
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def record_failure(self):
        with self._lock:
            self.failures += 1

    def estimated_cost_usd(self) -> float:
        rate = PRICING_PER_MTOK.get(active_model(), DEFAULT_PRICING)
        return round(
            self.prompt_tokens / 1e6 * rate["in"]
            + self.completion_tokens / 1e6 * rate["out"],
            6,
        )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "llm_calls": self.calls,
                "retries": self.retries,
                "failures": self.failures,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "llm_seconds": round(self.seconds, 2),
                "estimated_cost_usd": self.estimated_cost_usd(),
                "cost_basis": (
                    f"ESTIMATED from published per-token rates for "
                    f"{active_model() or 'n/a'}; not a billing record."
                ),
            }


METER = UsageMeter()


# ---------------------------------------------------------------- transport


class LLMError(RuntimeError):
    """Transport or provider error, with the provider's own message preserved."""


# Transient conditions worth retrying. A 400 (bad request) or 401 (bad key) is
# not transient — retrying it just wastes the user's time and hides the cause.
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with _Throttle():
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))


def _post_with_retries(url: str, payload: dict, headers: dict) -> tuple:
    """Returns (data, retries_used). Raises LLMError with the provider's message."""
    last = None
    for attempt in range(MAX_RETRIES):
        retry_after = None
        try:
            return _post_json(url, payload, headers), attempt
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                detail = ""
            last = LLMError(f"HTTP {e.code} from {url}: {detail}")
            if e.code not in RETRYABLE_STATUS:
                raise last from e
            if e.code == 429:
                # Prefer the provider's own instruction over our guess.
                try:
                    retry_after = float(e.headers.get("Retry-After") or 0) or None
                except (TypeError, ValueError):
                    retry_after = None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = LLMError(f"{type(e).__name__} calling {url}: {e}")
        except json.JSONDecodeError as e:
            last = LLMError(f"Provider returned non-JSON: {e}")

        if attempt < MAX_RETRIES - 1:
            # Exponential backoff with jitter, so a rate-limited cohort run does
            # not resend every case in lockstep and re-trigger the limit.
            delay = retry_after if retry_after else min(30.0, 1.0 * (2 ** attempt))
            time.sleep(delay * (0.6 + 0.8 * random.random()))

    METER.record_failure()
    raise last


def _call_openai_compatible(messages: list, tools: list = None, max_tokens: int = 400) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max(max_tokens, MIN_MAX_TOKENS),
        "temperature": 0.2,
        **EXTRA_BODY,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    t0 = time.time()
    data, retries = _post_with_retries(f"{BASE_URL}/chat/completions", payload, headers)
    METER.record(data.get("usage") or {}, time.time() - t0, retries)

    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"Provider returned no choices: {json.dumps(data)[:300]}")
    msg = dict(choices[0]["message"])
    # Reasoning models keep their chain of thought in a side channel. Drop it:
    # it is not an explanation a reviewer should read, and on a truncated
    # response it is all we would otherwise have.
    msg.pop("reasoning_content", None)
    if msg.get("content") is None:
        msg["content"] = ""
    return msg


def _call_anthropic(messages: list, tools: list = None, max_tokens: int = 400) -> dict:
    # Anthropic takes the system prompt as a top-level field, not a message.
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    convo = [m for m in messages if m.get("role") != "system"]

    raw_messages = [_to_anthropic_message(m) for m in convo]
    merged_messages = []
    for rm in raw_messages:
        if merged_messages and merged_messages[-1]["role"] == rm["role"] == "user":
            if isinstance(merged_messages[-1]["content"], list) and isinstance(rm["content"], list):
                merged_messages[-1]["content"].extend(rm["content"])
            elif isinstance(merged_messages[-1]["content"], str) and isinstance(rm["content"], list):
                merged_messages[-1]["content"] = [{"type": "text", "text": merged_messages[-1]["content"]}] + rm["content"]
            elif isinstance(merged_messages[-1]["content"], list) and isinstance(rm["content"], str):
                merged_messages[-1]["content"].append({"type": "text", "text": rm["content"]})
            else:
                merged_messages[-1]["content"] += "\n" + rm["content"]
        else:
            merged_messages.append(rm)

    payload = {
        "model": active_model(),
        # Same floor as the OpenAI-compatible path. Current Claude models think
        # by default, and thinking tokens count against max_tokens — a 200-token
        # ceiling can be consumed entirely before a single visible token is
        # emitted, leaving `content` empty. The floor was applied on one path
        # and not the other.
        "max_tokens": max(max_tokens, MIN_MAX_TOKENS),
        "messages": merged_messages,
    }
    if system:
        payload["system"] = system
    if tools:
        # Anthropic's tool schema differs from OpenAI's; convert it.
        payload["tools"] = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"].get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            for t in tools
        ]
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
    }

    t0 = time.time()
    data, retries = _post_with_retries(
        "https://api.anthropic.com/v1/messages", payload, headers)
    u = data.get("usage") or {}
    METER.record({"prompt_tokens": u.get("input_tokens"),
                  "completion_tokens": u.get("output_tokens")},
                 time.time() - t0, retries)

    # Normalise into the OpenAI message shape callers expect.
    text_parts, tool_calls = [], []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block["input"]),
                },
            })
    msg = {"role": "assistant", "content": "\n".join(text_parts).strip()}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _to_anthropic_message(m: dict) -> dict:
    """Translate one OpenAI-shaped message into Anthropic's content-block form."""
    role = m.get("role")
    if role == "tool":
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": m.get("content", ""),
            }],
        }
    if role == "assistant" and m.get("tool_calls"):
        blocks = []
        if m.get("content"):
            blocks.append({"type": "text", "text": m["content"]})
        for tc in m["tool_calls"]:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            blocks.append({"type": "tool_use", "id": tc.get("id", tc["function"]["name"]),
                           "name": tc["function"]["name"], "input": args})
        return {"role": "assistant", "content": blocks}
    return {"role": role, "content": m.get("content") or ""}


def chat(messages: list, tools: list = None, max_tokens: int = 400) -> dict:
    """
    Returns an OpenAI-shaped assistant message dict:
      {"role": "assistant", "content": str, "tool_calls": [...] (optional)}
    Raises LLMError on transport failure — callers decide whether to fall back.
    """
    if not is_enabled():
        raise LLMError("No LLM provider configured")
    if PROVIDER == "anthropic":
        return _call_anthropic(messages, tools, max_tokens)
    return _call_openai_compatible(messages, tools, max_tokens)


def reason_with_llm_or_fallback(task: str, context: str, fallback: str) -> str:
    """
    Single-turn helper for Mode 1 agents that only need a sentence of reasoning.

    Never raises: Mode 1 must produce the same DECISION with or without a
    provider, so a provider outage degrades the prose and nothing else.
    """
    if not (is_enabled() and NARRATION):
        return fallback
    try:
        msg = chat([{"role": "user", "content": f"{task}\n\nContext:\n{context}"}],
                   max_tokens=200)
        text = (msg.get("content") or "").strip()
        return text if text else fallback
    except Exception as e:
        return f"{fallback} [llm_error:{type(e).__name__}]"


def supports_tool_calling() -> tuple:
    """
    Probe whether the configured model actually emits tool calls.

    Modes 2/3 are meaningless on a model that ignores the tool schema and
    answers in prose, and providers do not advertise this reliably per model.
    Returns (ok: bool, detail: str).
    """
    probe = [{
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Call this to acknowledge. Always call it.",
            "parameters": {"type": "object",
                           "properties": {"value": {"type": "string"}},
                           "required": ["value"]},
        },
    }]
    try:
        msg = chat([{"role": "user", "content": "Call the ping tool with value='ok'."}],
                   tools=probe, max_tokens=200)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    if msg.get("tool_calls"):
        return True, f"emitted tool_call: {msg['tool_calls'][0]['function']['name']}"
    return False, f"no tool_calls; replied with text: {(msg.get('content') or '')[:120]!r}"
