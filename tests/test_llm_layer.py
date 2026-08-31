"""
Tests for the provider layer: retries, throttling, accounting, narration.

All offline. A fake HTTP transport stands in for the provider so retry and
rate-limit behaviour is tested deterministically — you cannot write a reliable
test for "does it back off correctly on 429" against a real API, because you
cannot make a real API 429 on demand.
"""
import io
import json
import os
import sys
import urllib.error

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from advanced import llm  # noqa: E402


def _http_error(code, retry_after=None):
    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    return urllib.error.HTTPError(
        "http://x", code, "err", headers, io.BytesIO(b'{"detail":"boom"}'))


@pytest.fixture
def fast_retries(monkeypatch):
    """Skip the real backoff sleeps; we are testing the policy, not the clock."""
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


@pytest.fixture
def openai_provider(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER", "openai_compatible")
    monkeypatch.setattr(llm, "BASE_URL", "http://fake")
    monkeypatch.setattr(llm, "MODEL", "fake-model")
    monkeypatch.setattr(llm, "API_KEY", "k")
    llm.METER.reset()


def _reply(content="hi", usage=None):
    return {"choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5}}


# --- retries ---------------------------------------------------------------

def test_retries_a_transient_failure_then_succeeds(monkeypatch, openai_provider, fast_retries):
    calls = []

    def flaky(url, payload, headers):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(503)
        return _reply()

    monkeypatch.setattr(llm, "_post_json", flaky)
    assert llm.chat([{"role": "user", "content": "x"}])["content"] == "hi"
    assert len(calls) == 3


def test_does_not_retry_a_bad_api_key(monkeypatch, openai_provider, fast_retries):
    """A 401 is not transient. Retrying it hides the cause and wastes the user's time."""
    calls = []

    def unauthorized(url, payload, headers):
        calls.append(1)
        raise _http_error(401)

    monkeypatch.setattr(llm, "_post_json", unauthorized)
    with pytest.raises(llm.LLMError, match="HTTP 401"):
        llm.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 1


def test_honours_retry_after_on_429(monkeypatch, openai_provider):
    slept, calls = [], []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))

    def limited(url, payload, headers):
        calls.append(1)
        if len(calls) < 2:
            raise _http_error(429, retry_after=7)
        return _reply()

    monkeypatch.setattr(llm, "_post_json", limited)
    llm.chat([{"role": "user", "content": "x"}])
    # Jitter is +/-40% around the provider's number, never an unrelated value.
    assert slept and 7 * 0.6 <= slept[0] <= 7 * 1.4


def test_gives_up_after_max_retries_and_counts_the_failure(
        monkeypatch, openai_provider, fast_retries):
    monkeypatch.setattr(llm, "_post_json",
                        lambda *a: (_ for _ in ()).throw(_http_error(503)))
    before = llm.METER.snapshot()["failures"]
    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "x"}])
    assert llm.METER.snapshot()["failures"] == before + 1


# --- accounting ------------------------------------------------------------

def test_usage_is_metered(monkeypatch, openai_provider):
    monkeypatch.setattr(llm, "_post_json", lambda *a: _reply(
        usage={"prompt_tokens": 100, "completion_tokens": 50}))
    llm.chat([{"role": "user", "content": "x"}])
    snap = llm.METER.snapshot()
    assert snap["prompt_tokens"] == 100 and snap["completion_tokens"] == 50
    assert snap["total_tokens"] == 150
    assert "ESTIMATED" in snap["cost_basis"]


def test_cost_uses_the_models_published_rate(monkeypatch, openai_provider):
    monkeypatch.setattr(llm, "MODEL", "gpt-4o-mini")
    monkeypatch.setattr(llm, "_post_json", lambda *a: _reply(
        usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}))
    llm.METER.reset()
    llm.chat([{"role": "user", "content": "x"}])
    rate = llm.PRICING_PER_MTOK["gpt-4o-mini"]
    assert llm.METER.estimated_cost_usd() == pytest.approx(rate["in"] + rate["out"])


# --- response normalisation ------------------------------------------------

def test_reasoning_traces_never_reach_the_caller(monkeypatch, openai_provider):
    """A paragraph of the model's deliberation is not an explanation for a reviewer."""
    monkeypatch.setattr(llm, "_post_json", lambda *a: {
        "choices": [{"message": {"role": "assistant", "content": "Answer.",
                                 "reasoning_content": "Hmm, let me think..."}}],
        "usage": {}})
    msg = llm.chat([{"role": "user", "content": "x"}])
    assert "reasoning_content" not in msg
    assert msg["content"] == "Answer."


def test_null_content_becomes_an_empty_string(monkeypatch, openai_provider):
    monkeypatch.setattr(llm, "_post_json", lambda *a: {
        "choices": [{"message": {"role": "assistant", "content": None}}], "usage": {}})
    assert llm.chat([{"role": "user", "content": "x"}])["content"] == ""


def test_empty_choices_raises_a_readable_error(monkeypatch, openai_provider):
    monkeypatch.setattr(llm, "_post_json", lambda *a: {"choices": [], "usage": {}})
    with pytest.raises(llm.LLMError, match="no choices"):
        llm.chat([{"role": "user", "content": "x"}])


# --- Anthropic translation -------------------------------------------------

def test_anthropic_lifts_system_out_of_the_message_list(monkeypatch):
    """Anthropic takes `system` as a top-level field; sending it as a message fails."""
    monkeypatch.setattr(llm, "PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "API_KEY", "k")
    monkeypatch.setattr(llm, "MODEL", "claude-sonnet-4-6")
    seen = {}

    def capture(url, payload, headers):
        seen.update(payload)
        return {"content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1}}

    monkeypatch.setattr(llm, "_post_json", capture)
    llm.chat([{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}])
    assert seen["system"] == "SYS"
    assert all(m["role"] != "system" for m in seen["messages"])


def test_anthropic_tool_results_become_content_blocks(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "API_KEY", "k")
    seen = {}
    monkeypatch.setattr(llm, "_post_json", lambda u, p, h: (
        seen.update(p), {"content": [{"type": "text", "text": "ok"}], "usage": {}})[1])
    llm.chat([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [
            {"id": "t1", "function": {"name": "f", "arguments": '{"a":1}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "42"},
    ])
    blocks = seen["messages"][1]["content"]
    assert blocks[0]["type"] == "tool_use" and blocks[0]["input"] == {"a": 1}
    assert seen["messages"][2]["content"][0]["type"] == "tool_result"


def test_anthropic_tool_calls_are_normalised_to_openai_shape(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "API_KEY", "k")
    monkeypatch.setattr(llm, "_post_json", lambda *a: {
        "content": [{"type": "tool_use", "id": "x", "name": "predict_risk",
                     "input": {"employee_id": "EMP4"}}], "usage": {}})
    msg = llm.chat([{"role": "user", "content": "x"}], tools=[])
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "predict_risk"
    assert json.loads(tc["function"]["arguments"]) == {"employee_id": "EMP4"}


# --- narration -------------------------------------------------------------

def test_narration_off_returns_the_fallback_without_calling_the_provider(
        monkeypatch, openai_provider):
    called = []
    monkeypatch.setattr(llm, "_post_json", lambda *a: called.append(1) or _reply())
    llm.set_narration(False)
    try:
        assert llm.reason_with_llm_or_fallback("t", "c", "FALLBACK") == "FALLBACK"
        assert called == []
    finally:
        llm.set_narration(True)


def test_a_provider_outage_degrades_prose_and_nothing_else(
        monkeypatch, openai_provider, fast_retries):
    """Mode 1 must reach the same decision when the provider is down."""
    monkeypatch.setattr(llm, "_post_json",
                        lambda *a: (_ for _ in ()).throw(_http_error(500)))
    llm.set_narration(True)
    out = llm.reason_with_llm_or_fallback("t", "c", "FALLBACK")
    assert out.startswith("FALLBACK")
    assert "llm_error" in out


# --- config ----------------------------------------------------------------

def test_every_preset_is_complete():
    for name, cfg in llm.PRESETS.items():
        assert cfg["model"], f"{name} has no default model"
        if name != "anthropic":
            assert cfg["base_url"].startswith("http"), f"{name} has no base_url"


def test_describe_never_contains_the_api_key(monkeypatch, openai_provider):
    monkeypatch.setattr(llm, "API_KEY", "sk-super-secret-value")
    assert "secret" not in llm.describe()


# --- Anthropic path: regressions from a previous round of fixes ------------

def test_anthropic_merges_consecutive_user_messages(monkeypatch):
    """
    Anthropic rejects two `user` turns in a row. The Mode 2/3 loop produces
    exactly that: a tool_result (which becomes role=user) followed by the
    "Call a tool, or call finalize" nudge.
    """
    monkeypatch.setattr(llm, "PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "API_KEY", "k")
    seen = {}
    monkeypatch.setattr(llm, "_post_json", lambda u, p, h: (
        seen.update(p), {"content": [{"type": "text", "text": "ok"}], "usage": {}})[1])

    llm.chat([
        {"role": "user", "content": "investigate"},
        {"role": "assistant", "tool_calls": [
            {"id": "t1", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "42"},
        {"role": "user", "content": "Call a tool, or call finalize to stop."},
    ])
    roles = [m["role"] for m in seen["messages"]]
    assert all(not (a == b == "user") for a, b in zip(roles, roles[1:])), roles


def test_anthropic_honours_the_max_tokens_floor(monkeypatch):
    """
    Current Claude models think by default and thinking counts against
    max_tokens, so a 200-token ceiling can be spent before any visible output.
    The floor was applied to the OpenAI-compatible path only.
    """
    monkeypatch.setattr(llm, "PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "API_KEY", "k")
    seen = {}
    monkeypatch.setattr(llm, "_post_json", lambda u, p, h: (
        seen.update(p), {"content": [{"type": "text", "text": "ok"}], "usage": {}})[1])
    llm.chat([{"role": "user", "content": "x"}], max_tokens=200)
    assert seen["max_tokens"] >= llm.MIN_MAX_TOKENS


def test_anthropic_preset_names_a_current_model():
    """
    A default nobody can call is worse than no default. Guards against both
    invented IDs and silently ageing ones.
    """
    model = llm.PRESETS["anthropic"]["model"]
    assert model.startswith("claude-")
    assert "claude-3" not in model, f"{model} is two generations old"
