#date: 2026-03-10T17:32:34Z
#url: https://api.github.com/gists/bf0bc43d974b8e6e39b630f8e5b36b2d
#owner: https://api.github.com/users/daniel-bernardes

# mypy: ignore-errors
"""
Diagnostic tests for the GPT-OSS "reasoning overflow" bug on Bedrock.

This bug manifests as:
  ValidationException: "**********"
  when GPT-OSS's internal reasoning chain overflows into Bedrock's response framing.

These tests run the SAME prompts across three setups to isolate the issue:
  1. GPT-OSS via Bedrock Converse API  (where the bug was observed)
  2. Claude via Bedrock Converse API    (control — same API, different model)
  3. GPT-OSS via Bedrock OpenAI-compat  (same model, different API)

Prerequisites:
  AWS credentials configured (e.g. `aws sso login`)

Run:
  uv run pytest tests/lighthouse/manual/test_fallback_reasoning_overflow.py -v -s
"""

import json
import os
import time

import boto3
import httpx
import pytest
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", os.environ.get("AWS_REGION", "us-west-2"))
GPT_OSS_MODEL = "openai.gpt-oss-120b-1:0"
CLAUDE_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"


def _has_aws_credentials() -> bool:
    try:
        session = boto3.Session(region_name=BEDROCK_REGION)
        creds = session.get_credentials()
        return creds is not None and creds.get_frozen_credentials().access_key is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_aws_credentials(),
    reason="Valid AWS credentials required",
)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bedrock_client():
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


class _SigV4Transport(httpx.BaseTransport):
    def __init__(self):
        self._transport = httpx.HTTPTransport()
        self._session = boto3.Session(region_name=BEDROCK_REGION)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        creds = self._session.get_credentials().get_frozen_credentials()
        sign_headers = {
            k: v for k, v in request.headers.items()
            if not k.lower().startswith("x-stainless")
            and k.lower() not in ("connection", "accept-encoding", "user-agent")
        }
        aws_req = AWSRequest(method=str(request.method), url=str(request.url),
                             data=request.content, headers=sign_headers)
        SigV4Auth(creds, "bedrock", BEDROCK_REGION).add_auth(aws_req)
        request.headers = httpx.Headers(dict(aws_req.headers))
        return self._transport.handle_request(request)

    def close(self):
        self._transport.close()


@pytest.fixture(scope="module")
def openai_compat_client():
    """OpenAI SDK client pointed at Bedrock's OpenAI-compatible endpoint."""
    return OpenAI(
        base_url=f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com/openai/v1",
        api_key="bedrock",
        http_client=httpx.Client(transport=_SigV4Transport()),
        timeout=90.0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_text(response) -> str:
    for block in response["output"]["message"]["content"]:
        if "text" in block:
            return block["text"]
    return ""


 "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"c "**********"a "**********"l "**********"l "**********"_ "**********"c "**********"o "**********"n "**********"v "**********"e "**********"r "**********"s "**********"e "**********"( "**********"b "**********"e "**********"d "**********"r "**********"o "**********"c "**********"k "**********"_ "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********", "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********", "**********"  "**********"m "**********"e "**********"s "**********"s "**********"a "**********"g "**********"e "**********"s "**********", "**********"  "**********"s "**********"y "**********"s "**********"t "**********"e "**********"m "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"= "**********"1 "**********"0 "**********"2 "**********"4 "**********") "**********": "**********"
    """Call Bedrock Converse API and return (text, error, elapsed, usage)."""
    t0 = time.monotonic()
    try:
        response = bedrock_client.converse(
            modelId=model,
            messages=messages,
            system=[{"text": system}],
            inferenceConfig={"maxTokens": "**********": 0.7},
        )
        elapsed = time.monotonic() - t0
        text = _get_text(response)
        usage = response.get("usage", {})
        return text, None, elapsed, usage
    except Exception as e:
        elapsed = time.monotonic() - t0
        return None, e, elapsed, {}


 "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"c "**********"a "**********"l "**********"l "**********"_ "**********"o "**********"p "**********"e "**********"n "**********"a "**********"i "**********"_ "**********"c "**********"o "**********"m "**********"p "**********"a "**********"t "**********"( "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********", "**********"  "**********"m "**********"e "**********"s "**********"s "**********"a "**********"g "**********"e "**********"s "**********"_ "**********"o "**********"p "**********"e "**********"n "**********"a "**********"i "**********", "**********"  "**********"s "**********"y "**********"s "**********"t "**********"e "**********"m "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"= "**********"1 "**********"0 "**********"2 "**********"4 "**********") "**********": "**********"
    """Call Bedrock OpenAI-compat endpoint and return (text, error, elapsed, usage)."""
    import re
    t0 = time.monotonic()
    try:
        msgs = [{"role": "system", "content": system}] + messages_openai
        response = client.chat.completions.create(
            model=GPT_OSS_MODEL,
            messages=msgs,
            max_completion_tokens= "**********"
        )
        elapsed = time.monotonic() - t0
        raw = response.choices[0].message.content or ""
        clean = re.sub(r"<reasoning>.*?</reasoning>", "", raw, flags=re.DOTALL).strip()
        usage = {"inputTokens": "**********": response.usage.completion_tokens}
        return clean, None, elapsed, usage
    except Exception as e:
        elapsed = time.monotonic() - t0
        return None, e, elapsed, {}


# ---------------------------------------------------------------------------
# Test prompts — systematically vary conversation length and complexity
# ---------------------------------------------------------------------------

SYSTEM_SUMMARIZE = "Summarize this coaching conversation in 2-3 sentences, highlighting the key insight the user had."
SYSTEM_COACH = "You are an empathetic life coach. Keep responses under 3 sentences."
SYSTEM_CLASSIFY = "Classify the user's overall sentiment in one word."

# 3-turn conversation (worked fine in previous tests)
MESSAGES_3_TURN = [
    {"role": "user", "content": [{"text": "I've been feeling overwhelmed at work."}]},
    {"role": "assistant", "content": [{"text": "What part of work feels most heavy right now?"}]},
    {"role": "user", "content": [{"text": "The constant meetings. I can never get deep work done."}]},
]

# 5-turn conversation (triggered the bug)
MESSAGES_5_TURN = [
    {"role": "user", "content": [{"text": "I want to work on my communication skills."}]},
    {"role": "assistant", "content": [{"text": "Communication is such a valuable skill. What specific aspect feels most important?"}]},
    {"role": "user", "content": [{"text": "Giving feedback to my direct reports. I tend to avoid difficult conversations."}]},
    {"role": "assistant", "content": [{"text": "It takes courage to recognize that. What happens when you think about those conversations?"}]},
    {"role": "user", "content": [{"text": "I worry about hurting their feelings or damaging the relationship."}]},
]

# 7-turn conversation (even longer)
MESSAGES_7_TURN = MESSAGES_5_TURN + [
    {"role": "assistant", "content": [{"text": "That concern shows you care deeply about your relationships. How do you think your reports feel when issues go unaddressed?"}]},
    {"role": "user", "content": [{"text": "They probably notice and wish I'd be more direct. I think the avoidance might actually be worse for the relationship."}]},
]

# 9-turn conversation
MESSAGES_9_TURN = MESSAGES_7_TURN + [
    {"role": "assistant", "content": [{"text": "That's a powerful insight. What would it look like to have one honest conversation this week?"}]},
    {"role": "user", "content": [{"text": "I could start with my most senior report. They'd probably appreciate the directness. Maybe I'll schedule a 1:1 specifically for feedback."}]},
]

# OpenAI-format versions for the compat endpoint
def _to_openai_format(msgs):
    return [{"role": m["role"], "content": m["content"][0]["text"]} for m in msgs]

OPENAI_3 = _to_openai_format(MESSAGES_3_TURN)
OPENAI_5 = _to_openai_format(MESSAGES_5_TURN)
OPENAI_7 = _to_openai_format(MESSAGES_7_TURN)
OPENAI_9 = _to_openai_format(MESSAGES_9_TURN)


# ===================================================================
# Test 1: Isolate by conversation length — GPT-OSS Converse API
# ===================================================================

class TestGptOssConverseByLength:
    """GPT-OSS via Bedrock Converse API at increasing conversation lengths."""

    def test_3_turn_summarize(self, bedrock_client):
        text, err, elapsed, usage = _call_converse(bedrock_client, GPT_OSS_MODEL, MESSAGES_3_TURN, SYSTEM_SUMMARIZE)
        print(f"\n[gpt_converse_3t] {elapsed:.2f}s | {usage}")
        print(f"  text: {text}")
        if err:
            print(f"  ERROR: {err}")
        assert err is None, f"3-turn should not fail: {err}"

    def test_5_turn_summarize(self, bedrock_client):
        text, err, elapsed, usage = _call_converse(bedrock_client, GPT_OSS_MODEL, MESSAGES_5_TURN, SYSTEM_SUMMARIZE)
        print(f"\n[gpt_converse_5t] {elapsed:.2f}s | {usage}")
        if text:
            print(f"  text: {text}")
        if err:
            print(f"  ERROR: {type(err).__name__}: {err}")

    def test_7_turn_summarize(self, bedrock_client):
        text, err, elapsed, usage = _call_converse(bedrock_client, GPT_OSS_MODEL, MESSAGES_7_TURN, SYSTEM_SUMMARIZE)
        print(f"\n[gpt_converse_7t] {elapsed:.2f}s | {usage}")
        if text:
            print(f"  text: {text}")
        if err:
            print(f"  ERROR: {type(err).__name__}: {err}")

    def test_9_turn_summarize(self, bedrock_client):
        text, err, elapsed, usage = _call_converse(bedrock_client, GPT_OSS_MODEL, MESSAGES_9_TURN, SYSTEM_SUMMARIZE)
        print(f"\n[gpt_converse_9t] {elapsed:.2f}s | {usage}")
        if text:
            print(f"  text: {text}")
        if err:
            print(f"  ERROR: {type(err).__name__}: {err}")

    def test_5_turn_coaching_response(self, bedrock_client):
        """Same 5 turns but different system prompt — is the bug prompt-specific?"""
        text, err, elapsed, usage = _call_converse(bedrock_client, GPT_OSS_MODEL, MESSAGES_5_TURN, SYSTEM_COACH)
        print(f"\n[gpt_converse_5t_coach] {elapsed:.2f}s | {usage}")
        if text:
            print(f"  text: {text}")
        if err:
            print(f"  ERROR: {type(err).__name__}: {err}")

    def test_5_turn_classify(self, bedrock_client):
        """Same 5 turns with a very short system prompt — minimal reasoning needed."""
        text, err, elapsed, usage = _call_converse(bedrock_client, GPT_OSS_MODEL, MESSAGES_5_TURN, SYSTEM_CLASSIFY)
        print(f"\n[gpt_converse_5t_classify] {elapsed:.2f}s | {usage}")
        if text:
            print(f"  text: {text}")
        if err:
            print(f"  ERROR: {type(err).__name__}: {err}")


# ===================================================================
# Test 2: Control — Claude via Bedrock Converse API (same API, different model)
# ===================================================================

class TestClaudeConverseByLength:
    """Claude via Bedrock Converse API — control group. Same API path, different model."""

    def test_3_turn_summarize(self, bedrock_client):
        text, err, elapsed, usage = _call_converse(bedrock_client, CLAUDE_MODEL, MESSAGES_3_TURN, SYSTEM_SUMMARIZE)
        print(f"\n[claude_converse_3t] {elapsed:.2f}s | {usage}")
        print(f"  text: {text}")
        assert err is None

    def test_5_turn_summarize(self, bedrock_client):
        text, err, elapsed, usage = _call_converse(bedrock_client, CLAUDE_MODEL, MESSAGES_5_TURN, SYSTEM_SUMMARIZE)
        print(f"\n[claude_converse_5t] {elapsed:.2f}s | {usage}")
        print(f"  text: {text}")
        assert err is None, f"Claude 5-turn should not fail: {err}"

    def test_7_turn_summarize(self, bedrock_client):
        text, err, elapsed, usage = _call_converse(bedrock_client, CLAUDE_MODEL, MESSAGES_7_TURN, SYSTEM_SUMMARIZE)
        print(f"\n[claude_converse_7t] {elapsed:.2f}s | {usage}")
        print(f"  text: {text}")
        assert err is None

    def test_9_turn_summarize(self, bedrock_client):
        text, err, elapsed, usage = _call_converse(bedrock_client, CLAUDE_MODEL, MESSAGES_9_TURN, SYSTEM_SUMMARIZE)
        print(f"\n[claude_converse_9t] {elapsed:.2f}s | {usage}")
        print(f"  text: {text}")
        assert err is None


# ===================================================================
# Test 3: GPT-OSS via OpenAI-compat endpoint (same model, different API)
# ===================================================================

class TestGptOssOpenAICompatByLength:
    """GPT-OSS via Bedrock OpenAI-compatible endpoint — same model, different API path."""

    def test_3_turn_summarize(self, openai_compat_client):
        text, err, elapsed, usage = _call_openai_compat(openai_compat_client, OPENAI_3, SYSTEM_SUMMARIZE)
        print(f"\n[gpt_openai_3t] {elapsed:.2f}s | {usage}")
        print(f"  text: {text}")
        assert err is None

    def test_5_turn_summarize(self, openai_compat_client):
        text, err, elapsed, usage = _call_openai_compat(openai_compat_client, OPENAI_5, SYSTEM_SUMMARIZE)
        print(f"\n[gpt_openai_5t] {elapsed:.2f}s | {usage}")
        if text:
            print(f"  text: {text}")
        if err:
            print(f"  ERROR: {type(err).__name__}: {err}")

    def test_7_turn_summarize(self, openai_compat_client):
        text, err, elapsed, usage = _call_openai_compat(openai_compat_client, OPENAI_7, SYSTEM_SUMMARIZE)
        print(f"\n[gpt_openai_7t] {elapsed:.2f}s | {usage}")
        if text:
            print(f"  text: {text}")
        if err:
            print(f"  ERROR: {type(err).__name__}: {err}")

    def test_9_turn_summarize(self, openai_compat_client):
        text, err, elapsed, usage = _call_openai_compat(openai_compat_client, OPENAI_9, SYSTEM_SUMMARIZE)
        print(f"\n[gpt_openai_9t] {elapsed:.2f}s | {usage}")
        if text:
            print(f"  text: {text}")
        if err:
            print(f"  ERROR: {type(err).__name__}: {err}")


# ===================================================================
# Test 4: "**********"
# ===================================================================

 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"T "**********"e "**********"s "**********"t "**********"G "**********"p "**********"t "**********"O "**********"s "**********"s "**********"C "**********"o "**********"n "**********"v "**********"e "**********"r "**********"s "**********"e "**********"M "**********"a "**********"x "**********"T "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
    """Test if increasing maxTokens prevents the reasoning overflow."""

    @pytest.mark.parametrize("max_tokens", [256, 512, 1024, 2048, 4096])
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"t "**********"e "**********"s "**********"t "**********"_ "**********"5 "**********"_ "**********"t "**********"u "**********"r "**********"n "**********"_ "**********"v "**********"a "**********"r "**********"y "**********"i "**********"n "**********"g "**********"_ "**********"m "**********"a "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"b "**********"e "**********"d "**********"r "**********"o "**********"c "**********"k "**********"_ "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********": "**********"
        text, err, elapsed, usage = _call_converse(
            bedrock_client, GPT_OSS_MODEL, MESSAGES_5_TURN, SYSTEM_SUMMARIZE,  "**********"= "**********"
        )
        status = "OK" if err is None else f"FAIL: {type(err).__name__}"
        print(f"\n[gpt_maxtkn_{max_tokens}] {status} | {elapsed: "**********"
        if text:
            print(f"  text: {text[:200]}")
        if err:
            # Extract the key part of the error
            err_str = str(err)
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"" "**********"u "**********"n "**********"e "**********"x "**********"p "**********"e "**********"c "**********"t "**********"e "**********"d "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"" "**********"  "**********"i "**********"n "**********"  "**********"e "**********"r "**********"r "**********"_ "**********"s "**********"t "**********"r "**********": "**********"
                print(f"  OVERFLOW BUG at maxTokens= "**********"
            else:
                print(f"  ERROR: {err_str[:200]}")


# ===================================================================
# Test 5: Tool calling with increasing conversation length
# ===================================================================

CLASSIFY_TOOL = {
    "toolSpec": {
        "name": "classify_sentiment",
        "description": "Classify the overall sentiment of the conversation.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed", "neutral"]},
                    "reasoning": {"type": "string"},
                },
                "required": ["sentiment", "reasoning"],
            }
        },
    }
}


class TestGptOssConverseToolsByLength:
    """Does the overflow bug also affect tool-calling requests?"""

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"c "**********"a "**********"l "**********"l "**********"_ "**********"w "**********"i "**********"t "**********"h "**********"_ "**********"t "**********"o "**********"o "**********"l "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"b "**********"e "**********"d "**********"r "**********"o "**********"c "**********"k "**********"_ "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********", "**********"  "**********"m "**********"e "**********"s "**********"s "**********"a "**********"g "**********"e "**********"s "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"= "**********"1 "**********"0 "**********"2 "**********"4 "**********") "**********": "**********"
        t0 = time.monotonic()
        try:
            response = bedrock_client.converse(
                modelId=GPT_OSS_MODEL,
                messages=messages,
                system=[{"text": "Classify the overall sentiment of this coaching conversation."}],
                toolConfig={"tools": [CLASSIFY_TOOL], "toolChoice": {"any": {}}},
                inferenceConfig={"maxTokens": "**********": 0.5},
            )
            elapsed = time.monotonic() - t0
            tool = None
            for block in response["output"]["message"]["content"]:
                if "toolUse" in block:
                    tool = block["toolUse"]
            return tool, None, elapsed
        except Exception as e:
            return None, e, time.monotonic() - t0

    def test_3_turn_tool(self, bedrock_client):
        tool, err, elapsed = self._call_with_tool(bedrock_client, MESSAGES_3_TURN)
        print(f"\n[gpt_tool_3t] {elapsed:.2f}s | {'OK' if err is None else f'FAIL: {err}'}")
        if tool:
            print(f"  tool: {json.dumps(tool['input'])}")
        assert err is None

    def test_5_turn_tool(self, bedrock_client):
        tool, err, elapsed = self._call_with_tool(bedrock_client, MESSAGES_5_TURN)
        print(f"\n[gpt_tool_5t] {elapsed:.2f}s | {'OK' if err is None else f'FAIL: {type(err).__name__}'}")
        if tool:
            print(f"  tool: {json.dumps(tool['input'])}")
        if err:
            print(f"  ERROR: {err}")

    def test_7_turn_tool(self, bedrock_client):
        tool, err, elapsed = self._call_with_tool(bedrock_client, MESSAGES_7_TURN)
        print(f"\n[gpt_tool_7t] {elapsed:.2f}s | {'OK' if err is None else f'FAIL: {type(err).__name__}'}")
        if tool:
            print(f"  tool: {json.dumps(tool['input'])}")
        if err:
            print(f"  ERROR: {err}")

    def test_9_turn_tool(self, bedrock_client):
        tool, err, elapsed = self._call_with_tool(bedrock_client, MESSAGES_9_TURN)
        print(f"\n[gpt_tool_9t] {elapsed:.2f}s | {'OK' if err is None else f'FAIL: {type(err).__name__}'}")
        if tool:
            print(f"  tool: {json.dumps(tool['input'])}")
        if err:
            print(f"  ERROR: {err}")


# ===================================================================
# Test 6: Reproducibility — run the failing case multiple times
# ===================================================================

class TestReproducibility:
    """Is the bug deterministic or intermittent?"""

    @pytest.mark.parametrize("attempt", range(5))
    def test_5_turn_summarize_repeated(self, bedrock_client, attempt):
        text, err, elapsed, usage = _call_converse(
            bedrock_client, GPT_OSS_MODEL, MESSAGES_5_TURN, SYSTEM_SUMMARIZE
        )
        status = "OK" if err is None else "FAIL"
        overflow = "**********"
        print(f"\n[repro_{attempt}] {status} | overflow={overflow} | {elapsed:.2f}s | {usage}")
        if text:
            print(f"  text: {text[:200]}")
