#date: 2026-03-10T17:32:34Z
#url: https://api.github.com/gists/bf0bc43d974b8e6e39b630f8e5b36b2d
#owner: https://api.github.com/users/daniel-bernardes

# mypy: ignore-errors
"""
Manual integration tests for the OpenAI → GPT-OSS fallback via Bedrock Converse API.

These tests hit the REAL Bedrock endpoint. They are NOT run in CI.
Prerequisites:
  1. AWS credentials configured (e.g. `aws sso login` or assumed role)

Run all direct Bedrock tests:
  uv run pytest tests/lighthouse/manual/test_fallback_integration.py -v -s

Run end-to-end through the actual gateway code:
  OPENAI_FALLBACK_ENABLED=true OPENAI_FALLBACK_FORCE=true \
    uv run pytest tests/lighthouse/manual/test_fallback_integration.py -v -s -k TestLlmGateway
"""

import json
import os
import time

import boto3
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", os.environ.get("AWS_REGION", "us-west-2"))
FALLBACK_MODEL = os.environ.get("OPENAI_FALLBACK_MODEL", "openai.gpt-oss-120b-1:0")


def _has_aws_credentials() -> bool:
    try:
        session = boto3.Session(region_name=BEDROCK_REGION)
        creds = session.get_credentials()
        return creds is not None and creds.get_frozen_credentials().access_key is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_aws_credentials(),
    reason="Valid AWS credentials required (e.g. run `aws sso login` first)",
)


@pytest.fixture(scope="module")
def bedrock_client():
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_text(response) -> str:
    for block in response["output"]["message"]["content"]:
        if "text" in block:
            return block["text"]
    return ""


def _get_tool_use(response) -> dict | None:
    for block in response["output"]["message"]["content"]:
        if "toolUse" in block:
            return block["toolUse"]
    return None


def _has_reasoning(response) -> bool:
    return any("reasoningContent" in block for block in response["output"]["message"]["content"])


# ===================================================================
# 1. BASIC CHAT — mirrors Behavior.generate_next_turns() patterns
# ===================================================================

class TestBasicChat:
    """Tests matching Lighthouse behaviors that call call_chat_completion_with_functions
    WITHOUT tools — pure text chat (32 call sites)."""

    def test_simple_response(self, bedrock_client):
        """Simplest case: one user message, one response."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "What is 2 + 2? Reply with just the number."}]}],
            inferenceConfig={"maxTokens": "**********": 0.5},
        )
        text = _get_text(response)
        print(f"\n[simple] Text: {text}")
        assert "4" in text

    def test_coaching_multi_turn(self, bedrock_client):
        """Multi-turn coaching conversation — mirrors orchestration behavior."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[
                {"role": "user", "content": [{"text": "I've been feeling overwhelmed at work."}]},
                {"role": "assistant", "content": [{"text": "I hear you — feeling overwhelmed is really tough. What part of work feels most heavy right now?"}]},
                {"role": "user", "content": [{"text": "The constant meetings. I can never get deep work done."}]},
            ],
            system=[{"text": "You are an empathetic life coach. Keep responses under 3 sentences."}],
            inferenceConfig={"maxTokens": "**********": 0.5},
        )
        text = _get_text(response)
        print(f"\n[multi_turn] Text: {text}")
        assert len(text) > 10

    def test_temperature_zero(self, bedrock_client):
        """Temperature=0 — mirrors OneWord behavior."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "I just got promoted and I can't stop smiling!"}]}],
            system=[{"text": "Reply with exactly one word that describes the user's mood."}],
            inferenceConfig={"maxTokens": "**********": 0},
        )
        text = _get_text(response)
        print(f"\n[temp_zero] Text: {text}")
        assert len(text.split()) <= 3

    def test_long_system_prompt(self, bedrock_client):
        """Long system prompt — mirrors real coaching prompts which can be 2000+ chars."""
        system_prompt = (
            "You are an expert life coach specializing in personal development. "
            "Your coaching style is warm, empathetic, and action-oriented. "
            "You use active listening techniques and ask powerful open-ended questions. "
            "You help clients identify their strengths, overcome obstacles, and create actionable plans. "
            "Always respond in a supportive, non-judgmental tone. "
            "Keep your responses concise — no more than 3 sentences per turn. "
            "Focus on the client's emotions first before offering strategies. "
            "If the client mentions a specific challenge, help them break it down into manageable steps. "
            "Remember previous context in the conversation and reference it when relevant. "
        ) * 3  # ~900 chars * 3 = ~2700 chars
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "I want to be a better leader but I don't know where to start."}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"maxTokens": "**********": 0.7},
        )
        text = _get_text(response)
        print(f"\n[long_prompt] Text: {text}")
        assert len(text) > 10

    def test_greeting_with_low_temperature(self, bedrock_client):
        """Temperature=0.1 — mirrors Pleasantries behavior."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "Hey there, it's been a tough week."}]}],
            system=[{"text": "You are a warm, supportive coach. Greet the user and acknowledge their feelings in 1-2 sentences."}],
            inferenceConfig={"maxTokens": "**********": 0.1},
        )
        text = _get_text(response)
        print(f"\n[greeting] Text: {text}")
        assert len(text) > 5

    @pytest.mark.xfail(reason="GPT-OSS Bedrock bug: reasoning overflows into internal headers on longer conversations")
    def test_conversation_wrapup(self, bedrock_client):
        """Wrap-up generation — mirrors ConversationRpg.generate_wrapup()."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[
                {"role": "user", "content": [{"text": "I want to work on my communication skills."}]},
                {"role": "assistant", "content": [{"text": "Communication is such a valuable skill to develop. What specific aspect feels most important to you right now?"}]},
                {"role": "user", "content": [{"text": "I think it's giving feedback to my direct reports. I tend to avoid difficult conversations."}]},
                {"role": "assistant", "content": [{"text": "It takes courage to recognize that. Many leaders struggle with this. What happens when you think about having those conversations?"}]},
                {"role": "user", "content": [{"text": "I worry about hurting their feelings or damaging the relationship."}]},
            ],
            system=[{"text": "Summarize this coaching conversation in 2-3 sentences, highlighting the key insight the user had."}],
            inferenceConfig={"maxTokens": "**********": 0.7},
        )
        text = _get_text(response)
        print(f"\n[wrapup] Text: {text}")
        assert len(text) > 20


# ===================================================================
# 2. REASONING SEPARATION
# ===================================================================

class TestReasoningSeparation:

    def test_reasoning_in_separate_block(self, bedrock_client):
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "What is 15 * 23?"}]}],
            inferenceConfig={"maxTokens": "**********"
        )
        block_types = [list(b.keys())[0] for b in response["output"]["message"]["content"]]
        print(f"\n[reasoning] Block types: {block_types}")
        assert _has_reasoning(response), "Expected reasoningContent block"
        text = _get_text(response)
        assert "<reasoning>" not in text, "Text should NOT contain reasoning tags"

    def test_text_block_is_clean(self, bedrock_client):
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "Explain photosynthesis in one sentence."}]}],
            inferenceConfig={"maxTokens": "**********"
        )
        text = _get_text(response)
        print(f"\n[clean_text] Text: {text}")
        assert "<reasoning>" not in text and "</reasoning>" not in text


# ===================================================================
# 3. TOOL CALLING — mirrors the 28 real tool-calling call sites
# ===================================================================

# Pattern A: Preflight classification (crisis_topic_detection, off_topic_guardrails, etc.)
CLASSIFY_TOOL = {
    "toolSpec": {
        "name": "classify_message",
        "description": "Classify the user message into a category with reasoning.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "classification": {"type": "string", "enum": ["on_topic", "off_topic", "crisis"]},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reasoning": {"type": "string", "description": "Brief explanation of the classification"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Evidence from the message supporting the classification",
                    },
                },
                "required": ["classification", "severity", "reasoning"],
            }
        },
    }
}

# Pattern B: Reward insight generation (reward_function_call.py)
INSIGHT_TOOL = {
    "toolSpec": {
        "name": "generate_insight",
        "description": "Generate a coaching insight based on the conversation.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Less than 5-sentence summary of the conversation"},
                    "generate_result": {"type": "string", "description": "A key coaching insight from the conversation"},
                },
                "required": ["summary", "generate_result"],
            }
        },
    }
}

# Pattern C: Top strength extraction (reward_types/top_strength)
STRENGTH_TOOL = {
    "toolSpec": {
        "name": "generate_insight",
        "description": "Extract the user's top coaching strength from the conversation.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Less than 5-sentence summary"},
                    "strength": {"type": "string", "description": "A key coaching strength category"},
                    "strength_insight": {"type": "string", "description": "An insight about the identified strength"},
                },
                "required": ["strength", "strength_insight"],
            }
        },
    }
}

# Pattern D: Schedule follow-up (behaviors/schedule_follow_up)
SCHEDULE_TOOL = {
    "toolSpec": {
        "name": "schedule_follow_up",
        "description": "Finalize the follow up date if the user accepts.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "follow_up_date": {
                        "type": "string",
                        "format": "date-time",
                        "description": "The date and time of the follow up, or nothing if they don't want one",
                    }
                },
                "required": [],
            }
        },
    }
}

# Pattern E: Action item identification (post_processors/identify_action_items)
ACTION_ITEM_TOOL = {
    "toolSpec": {
        "name": "identify_action_items",
        "description": "Identify action items from the coaching conversation.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "has_action_items": {"type": "boolean", "description": "Whether action items were identified"},
                    "action_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "due_date": {"type": "string"},
                            },
                        },
                        "description": "List of action items",
                    },
                },
                "required": ["has_action_items"],
            }
        },
    }
}

# Pattern F: Commitment detection (preflight_checks/commitment_handler)
COMMITMENT_TOOL = {
    "toolSpec": {
        "name": "detect_commitment",
        "description": "Detect if the user has made a commitment or expressed intent to take action.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "has_commitment": {"type": "boolean"},
                    "commitment_text": {"type": "string", "description": "The commitment the user made"},
                    "confidence": {"type": "number", "description": "Confidence score 0-1"},
                },
                "required": ["has_commitment"],
            }
        },
    }
}


class TestToolCalling:
    """Tests matching real Lighthouse tool-calling patterns."""

    # -- Pattern A: Preflight classification --

    def test_classify_on_topic(self, bedrock_client):
        """On-topic coaching message classification."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "I'd like to work on my leadership skills and become a better manager."}]}],
            system=[{"text": "You are a message classifier for a coaching app. Classify the user's message."}],
            toolConfig={"tools": [CLASSIFY_TOOL], "toolChoice": {"auto": {}}},
            inferenceConfig={"maxTokens": "**********"
        )
        tool = _get_tool_use(response)
        print(f"\n[classify_on_topic] {json.dumps(tool['input'], indent=2) if tool else 'NO TOOL'}")
        assert tool is not None
        assert tool["input"]["classification"] == "on_topic"

    def test_classify_off_topic(self, bedrock_client):
        """Off-topic message classification."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "What's the weather like in San Francisco today?"}]}],
            system=[{"text": "You are a message classifier for a coaching app. Messages about weather, sports, or unrelated topics are off_topic."}],
            toolConfig={"tools": [CLASSIFY_TOOL], "toolChoice": {"auto": {}}},
            inferenceConfig={"maxTokens": "**********"
        )
        tool = _get_tool_use(response)
        print(f"\n[classify_off_topic] {json.dumps(tool['input'], indent=2) if tool else 'NO TOOL'}")
        assert tool is not None
        assert tool["input"]["classification"] == "off_topic"

    def test_classify_crisis(self, bedrock_client):
        """Crisis detection classification."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "I don't see the point in going on anymore. Everything feels hopeless."}]}],
            system=[{"text": "You are a message classifier for a coaching app. Messages expressing self-harm, suicidal ideation, or severe distress should be classified as crisis with severity high."}],
            toolConfig={"tools": [CLASSIFY_TOOL], "toolChoice": {"any": {}}},
            inferenceConfig={"maxTokens": "**********"
        )
        tool = _get_tool_use(response)
        print(f"\n[classify_crisis] {json.dumps(tool['input'], indent=2) if tool else 'NO TOOL'}")
        assert tool is not None
        assert tool["input"]["classification"] == "crisis"
        assert tool["input"]["severity"] == "high"

    def test_classify_with_named_tool_choice(self, bedrock_client):
        """Named tool choice — some call sites use this pattern."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "Can you help me with my resume?"}]}],
            system=[{"text": "Classify the message."}],
            toolConfig={"tools": [CLASSIFY_TOOL], "toolChoice": {"tool": {"name": "classify_message"}}},
            inferenceConfig={"maxTokens": "**********"
        )
        tool = _get_tool_use(response)
        print(f"\n[named_choice] {json.dumps(tool['input'], indent=2) if tool else 'NO TOOL'}")
        assert tool is not None
        assert tool["name"] == "classify_message"

    # -- Pattern B: Reward insight generation --

    def test_generate_insight(self, bedrock_client):
        """Reward function insight generation with multi-turn coaching transcript."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[
                {"role": "user", "content": [{"text": "I've been struggling with work-life balance."}]},
                {"role": "assistant", "content": [{"text": "What does an ideal balance look like for you?"}]},
                {"role": "user", "content": [{"text": "I want to be present for my kids' activities but I keep bringing work home."}]},
                {"role": "assistant", "content": [{"text": "It sounds like being present for your family is a core value. What's one boundary you could set this week?"}]},
                {"role": "user", "content": [{"text": "I could commit to no laptops after 6pm on weeknights."}]},
            ],
            system=[{"text": "Use this transcript of a coaching session to extract insights."}],
            toolConfig={"tools": [INSIGHT_TOOL], "toolChoice": {"auto": {}}},
            inferenceConfig={"maxTokens": "**********": 0.7},
        )
        tool = _get_tool_use(response)
        print(f"\n[insight] {json.dumps(tool['input'], indent=2) if tool else 'NO TOOL'}")
        assert tool is not None
        assert "summary" in tool["input"]
        assert "generate_result" in tool["input"]
        assert len(tool["input"]["generate_result"]) > 20

    # -- Pattern C: Top strength extraction --

    def test_extract_strength(self, bedrock_client):
        """Top strength reward type with required fields."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[
                {"role": "user", "content": [{"text": "I delegated the sprint planning to my junior dev and it went really well. I was nervous but glad I did it."}]},
            ],
            system=[{"text": "Extract the user's top coaching strength from this conversation."}],
            toolConfig={"tools": [STRENGTH_TOOL], "toolChoice": {"auto": {}}},
            inferenceConfig={"maxTokens": "**********": 0.7},
        )
        tool = _get_tool_use(response)
        print(f"\n[strength] {json.dumps(tool['input'], indent=2) if tool else 'NO TOOL'}")
        assert tool is not None
        assert "strength" in tool["input"]
        assert "strength_insight" in tool["input"]
        assert len(tool["input"]["strength_insight"]) > 20

    # -- Pattern D: Schedule follow-up --

    def test_schedule_with_date(self, bedrock_client):
        """Schedule follow-up with a date — mirrors schedule_follow_up behavior."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[
                {"role": "assistant", "content": [{"text": "Would you like to schedule a follow-up?"}]},
                {"role": "user", "content": [{"text": "Yes, let's do next Tuesday at 3pm."}]},
            ],
            system=[{"text": "Return a follow up date if one has been given in the last few messages, otherwise an empty string."}],
            toolConfig={"tools": [SCHEDULE_TOOL], "toolChoice": {"auto": {}}},
            inferenceConfig={"maxTokens": "**********"
        )
        tool = _get_tool_use(response)
        print(f"\n[schedule] {json.dumps(tool['input'], indent=2) if tool else 'NO TOOL'}")
        assert tool is not None
        assert "follow_up_date" in tool["input"]
        assert len(tool["input"]["follow_up_date"]) > 0

    def test_schedule_declined(self, bedrock_client):
        """User declines follow-up — tool should return empty or no date."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[
                {"role": "assistant", "content": [{"text": "Would you like to schedule a follow-up?"}]},
                {"role": "user", "content": [{"text": "No thanks, I'm good for now."}]},
            ],
            system=[{"text": "Return a follow up date if one has been given, otherwise an empty string."}],
            toolConfig={"tools": [SCHEDULE_TOOL], "toolChoice": {"auto": {}}},
            inferenceConfig={"maxTokens": "**********"
        )
        tool = _get_tool_use(response)
        print(f"\n[schedule_declined] {json.dumps(tool['input'], indent=2) if tool else 'NO TOOL'}")
        # Tool may or may not be called; if called, date should be empty
        if tool:
            assert tool["input"].get("follow_up_date", "") == ""

    # -- Pattern E: Action item identification --

    def test_identify_action_items(self, bedrock_client):
        """Post-processor action item identification."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[
                {"role": "user", "content": [{"text": "I'm going to start meditating for 10 minutes every morning and journal before bed."}]},
            ],
            system=[{"text": "Identify any action items the user committed to in this conversation."}],
            toolConfig={"tools": [ACTION_ITEM_TOOL], "toolChoice": {"any": {}}},
            inferenceConfig={"maxTokens": "**********"
        )
        tool = _get_tool_use(response)
        print(f"\n[action_items] {json.dumps(tool['input'], indent=2) if tool else 'NO TOOL'}")
        assert tool is not None
        assert tool["input"]["has_action_items"] is True
        assert len(tool["input"].get("action_items", [])) >= 1

    # -- Pattern F: Commitment detection --

    def test_detect_commitment(self, bedrock_client):
        """Commitment handler preflight check."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[
                {"role": "user", "content": [{"text": "I'm definitely going to have that difficult conversation with my manager this week."}]},
            ],
            system=[{"text": "Detect if the user has made a commitment or expressed intent to take action."}],
            toolConfig={"tools": [COMMITMENT_TOOL], "toolChoice": {"any": {}}},
            inferenceConfig={"maxTokens": "**********"
        )
        tool = _get_tool_use(response)
        print(f"\n[commitment] {json.dumps(tool['input'], indent=2) if tool else 'NO TOOL'}")
        assert tool is not None
        assert tool["input"]["has_commitment"] is True

    # -- Tool argument quality --

    def test_tool_arguments_are_clean(self, bedrock_client):
        """Tool call arguments should not contain reasoning tags."""
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "I gave my coworker constructive feedback and it went well."}]}],
            system=[{"text": "Extract the user's top coaching strength."}],
            toolConfig={"tools": [STRENGTH_TOOL], "toolChoice": {"any": {}}},
            inferenceConfig={"maxTokens": "**********"
        )
        tool = _get_tool_use(response)
        assert tool is not None
        raw = json.dumps(tool["input"])
        print(f"\n[clean_args] {raw}")
        assert "<reasoning>" not in raw


# ===================================================================
# 4. LATENCY
# ===================================================================

class TestLatency:

    def test_simple_response_latency(self, bedrock_client):
        t0 = time.monotonic()
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "How are you?"}]}],
            system=[{"text": "Reply in one sentence."}],
            inferenceConfig={"maxTokens": "**********"
        )
        elapsed = time.monotonic() - t0
        print(f"\n[latency_chat] {elapsed:.2f}s — {_get_text(response)}")
        print(f"  Usage: {response['usage']}")

    def test_tool_calling_latency(self, bedrock_client):
        t0 = time.monotonic()
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[{"role": "user", "content": [{"text": "I want to improve my time management."}]}],
            system=[{"text": "Classify this coaching message."}],
            toolConfig={"tools": [CLASSIFY_TOOL], "toolChoice": {"auto": {}}},
            inferenceConfig={"maxTokens": "**********"
        )
        elapsed = time.monotonic() - t0
        tool = _get_tool_use(response)
        print(f"\n[latency_tool] {elapsed:.2f}s — {json.dumps(tool['input']) if tool else 'NO TOOL'}")
        print(f"  Usage: {response['usage']}")

    def test_multi_turn_with_tools_latency(self, bedrock_client):
        """Realistic latency: multi-turn conversation + tool calling."""
        t0 = time.monotonic()
        response = bedrock_client.converse(
            modelId=FALLBACK_MODEL,
            messages=[
                {"role": "user", "content": [{"text": "I've been working on delegating more."}]},
                {"role": "assistant", "content": [{"text": "That's great progress. How has it been going?"}]},
                {"role": "user", "content": [{"text": "I let my junior dev lead sprint planning last week and it went really well."}]},
                {"role": "assistant", "content": [{"text": "That sounds like a great experience. What did you learn?"}]},
                {"role": "user", "content": [{"text": "I realized I don't need to control everything."}]},
            ],
            system=[{"text": "Use this transcript to extract a coaching insight."}],
            toolConfig={"tools": [INSIGHT_TOOL], "toolChoice": {"auto": {}}},
            inferenceConfig={"maxTokens": "**********": 0.7},
        )
        elapsed = time.monotonic() - t0
        tool = _get_tool_use(response)
        print(f"\n[latency_multi] {elapsed:.2f}s")
        if tool:
            print(f"  Insight: {json.dumps(tool['input'], indent=2)}")
        print(f"  Usage: {response['usage']}")


# ===================================================================
# 5. END-TO-END through llm_gateway (with force-fallback)
# ===================================================================

class TestLlmGatewayEndToEnd:
    """Test through the actual llm_gateway fallback code.

    These exercise _call_openai_with_functions → _bedrock_converse_fallback,
    matching the exact calling conventions of real Lighthouse code.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_force(self):
        if os.environ.get("OPENAI_FALLBACK_FORCE", "").lower() != "true":
            pytest.skip("Set OPENAI_FALLBACK_ENABLED=true OPENAI_FALLBACK_FORCE=true")

    def test_e2e_no_tools_coaching_chat(self):
        """Behavior.generate_next_turns() pattern — no tools, just chat."""
        from lighthouse.gateways.llm_gateway import _call_openai_with_functions

        result = _call_openai_with_functions(
            functions=[],
            messages=[{"isSystem": False, "text": "I want to improve my communication skills."}],
            model="gpt-4o",
            prompt="You are a helpful life coach. Respond in 1-2 sentences.",
            temperature=0.7,
            tool_choice=None,
        )
        print(f"\n[e2e_chat] {result}")
        assert len(result) > 0
        assert len(result[0]["response_text"]) > 10

    def test_e2e_reward_function_call_pattern(self):
        """Exact pattern from reward_function_call.py — tool_choice=auto, single tool."""
        from lighthouse.gateways.llm_gateway import call_chat_completion_with_functions

        result = call_chat_completion_with_functions(
            prompt="Use this transcript of a coaching session to extract insights",
            messages=[
                {"isSystem": False, "text": "I've been working on my confidence."},
                {"isSystem": True, "text": "What specific situations challenge your confidence?"},
                {"isSystem": False, "text": "Speaking up in team meetings. I always think my ideas aren't good enough."},
            ],
            model="gpt-4o",
            temperature=0.7,
            functions=[{
                "name": "generate_insight",
                "description": "A key coaching insight from the conversation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Less than 5-sentence summary"},
                        "generate_result": {"type": "string", "description": "A key coaching insight"},
                    },
                },
            }],
            tool_choice="auto",
        )
        print(f"\n[e2e_reward] {result}")
        assert len(result) > 0
        # Should have tool_calls in the response (matching reward_function_call.py parsing)
        if result[0].get("tool_calls"):
            fn = result[0]["tool_calls"][0]["function"]
            args = json.loads(fn.arguments)
            print(f"  Parsed insight: {json.dumps(args, indent=2)}")
            assert "generate_result" in args or "summary" in args

    def test_e2e_schedule_follow_up_pattern(self):
        """Exact pattern from schedule_follow_up behavior."""
        from lighthouse.gateways.llm_gateway import call_chat_completion_with_functions

        result = call_chat_completion_with_functions(
            prompt="Return a follow up date if one has been given in the last few messages, otherwise an empty string",
            messages=[
                {"isSystem": True, "text": "Would you like to schedule a follow-up?"},
                {"isSystem": False, "text": "Yes, let's do next Wednesday at 2pm."},
            ],
            model="gpt-4o",
            temperature=0.7,
            functions=[{
                "name": "schedule_follow_up",
                "description": "Finalize the follow up date if the user accepts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "follow_up_date": {
                            "type": "string",
                            "format": "date-time",
                            "description": "The date and time of the follow up",
                        }
                    },
                    "required": [],
                },
            }],
            tool_choice="auto",
        )
        print(f"\n[e2e_schedule] {result}")
        assert len(result) > 0
        if result[0].get("tool_calls"):
            fn = result[0]["tool_calls"][0]["function"]
            args = json.loads(fn.arguments)
            print(f"  Schedule args: {args}")

    def test_e2e_no_tools_multi_turn(self):
        """Multi-turn conversation without tools — mirrors most behavior patterns."""
        from lighthouse.gateways.llm_gateway import call_chat_completion_with_functions

        result = call_chat_completion_with_functions(
            prompt="You are an empathetic life coach. Keep responses under 3 sentences.",
            messages=[
                {"isSystem": False, "text": "I've been feeling overwhelmed at work."},
                {"isSystem": True, "text": "That sounds really challenging. What part feels most heavy?"},
                {"isSystem": False, "text": "The constant context switching between projects."},
            ],
            model="gpt-4o",
            temperature=0.5,
        )
        print(f"\n[e2e_multi] {result}")
        assert len(result) > 0
        assert len(result[0]["response_text"]) > 10

    def test_e2e_strength_extraction_pattern(self):
        """Top strength reward type pattern — multiple required fields."""
        from lighthouse.gateways.llm_gateway import call_chat_completion_with_functions

        result = call_chat_completion_with_functions(
            prompt="Extract the user's top coaching strength from this conversation.",
            messages=[
                {"isSystem": False, "text": "I gave my teammate really honest feedback about their presentation and they thanked me for it."},
            ],
            model="gpt-4o",
            temperature=0.7,
            functions=[{
                "name": "generate_insight",
                "description": "Extract the user's top coaching strength",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Summary"},
                        "strength": {"type": "string", "description": "Key strength category"},
                        "strength_insight": {"type": "string", "description": "Insight about the strength"},
                    },
                    "required": ["strength", "strength_insight"],
                },
            }],
            tool_choice="auto",
        )
        print(f"\n[e2e_strength] {result}")
        assert len(result) > 0
        if result[0].get("tool_calls"):
            fn = result[0]["tool_calls"][0]["function"]
            args = json.loads(fn.arguments)
            print(f"  Strength: {args.get('strength')}")
            print(f"  Insight: {args.get('strength_insight')}")
            assert "strength" in args
