#date: 2025-12-09T17:07:09Z
#url: https://api.github.com/gists/220038b4a0ef16cb5357827788d43213
#owner: https://api.github.com/users/graelo

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mistral tool call parser for v11+ models.

This implementation uses token-based parsing for streaming, leveraging the
atomic nature of special token IDs ([TOOL_CALLS], [ARGS], [CALL_ID]) to
reliably detect tool call boundaries.

Supported models: Mistral-Small-3.1+, Ministral-3+, and other v11+ models.

Note: Pre-v11 models (Mistral-7B-Instruct-v0.1/v0.2/v0.3) are not supported.
These older models have limited tool calling capabilities and require complex
text-based parsing with partial JSON handling. Users should upgrade to v11+
models for reliable tool calling support.
"""

import contextlib
from collections.abc import Sequence
from enum import Enum, auto
from random import choices
from string import ascii_letters, digits

import regex as re
from pydantic import Field

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.logger import init_logger
from vllm.tokenizers import MistralTokenizer, TokenizerLike

logger = init_logger(__name__)

ALPHANUMERIC = ascii_letters + digits


def _escape_json_control_chars(s: str) -> str:
    """Escape control characters that would break JSON serialization.

    Models sometimes emit raw control characters (literal newlines, tabs, etc.)
    inside JSON strings. These must be escaped for valid JSON output.
    Already-escaped sequences (like the two-char '\\n') are left untouched.
    """
    return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


class MistralToolCall(ToolCall):
    id: str = Field(default_factory=lambda: MistralToolCall.generate_random_id())

    @staticmethod
    def generate_random_id():
        # Mistral Tool Call Ids must be alphanumeric with a length of 9.
        # https://github.com/mistralai/mistral-common/blob/21ee9f6cee3441e9bb1e6ed2d10173f90bd9b94b/src/mistral_common/protocol/instruct/validator.py#L299
        return "".join(choices(ALPHANUMERIC, k=9))

    @staticmethod
    def is_valid_id(id: str) -> bool:
        return id.isalnum() and len(id) == 9


class StreamingState(Enum):
    """Streaming state for tool call parsing."""

    CONTENT = "**********"
    PARSING_TOOL_NAME = auto()  # After [TOOL_CALLS], parsing function name
    PARSING_TOOL_ARGS = auto()  # Parsing JSON arguments
    COMPLETE = auto()  # All tools parsed


class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral v11+ models.

    Supports the v11+ format: [TOOL_CALLS]name[ARGS]{...}
    Optionally with call ID: [TOOL_CALLS]name[CALL_ID]id[ARGS]{...}

    This parser requires MistralTokenizer (tokenizer_mode= "**********"
    models using tokenizer version 11 or higher.
    """

    def __init__(self, tokenizer: "**********":
        super().__init__(tokenizer)

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"s "**********"i "**********"n "**********"s "**********"t "**********"a "**********"n "**********"c "**********"e "**********"( "**********"s "**********"e "**********"l "**********"f "**********". "**********"m "**********"o "**********"d "**********"e "**********"l "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********", "**********"  "**********"M "**********"i "**********"s "**********"t "**********"r "**********"a "**********"l "**********"T "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********") "**********": "**********"
            raise RuntimeError(
                "MistralToolParser requires MistralTokenizer. "
                "Please use tokenizer_mode= "**********"
                "Note: Only v11+ Mistral models are supported for tool calling."
            )

        self._mistral_base_tokenizer = "**********"
        self._version = "**********"

        if self._version < 11:
            raise RuntimeError(
                f"MistralToolParser requires tokenizer version 11 or higher, "
                f"but got version {self._version}. Pre-v11 models "
                "(Mistral-7B-Instruct-v0.1/v0.2/v0.3) are not supported for "
                "tool calling. Please use a v11+ model such as "
                "Mistral-Small-3.1 or Ministral-3."
            )

        # Get bot token info
        self.bot_token = "**********"
        self.bot_token_id = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"b "**********"o "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
            raise RuntimeError(
                "Mistral Tool Parser could not locate the [TOOL_CALLS] token "
                "in the tokenizer!"
            )

        # Get control tokens for v11+ format
        try:
            self._args_token_id = "**********"
                "[ARGS]"
            )
        except Exception as err:
            raise RuntimeError(
                "Mistral Tool Parser could not locate the [ARGS] token. "
                "This token is required for v11+ tool call parsing."
            ) from err

        self._call_id_token_id: "**********"
        with contextlib.suppress(Exception):
            # [CALL_ID] is optional - some models may not have it
            self._call_id_token_id = "**********"
                "[CALL_ID]"
            )

        # Regex for non-streaming parsing: name{args}
        self.fn_name_regex = re.compile(r"([a-zA-Z0-9_-]+)(\{[\s\S]*?\}+)", re.DOTALL)

        # Streaming state
        self._streaming_state = StreamingState.CONTENT
        self._current_tool_index = -1
        self._current_tool_id: str | None = None
        self._current_tool_name: str = ""
        self._current_tool_args: str = ""
        self._brace_depth = 0

        # For compatibility with serving_chat.py's finish_reason detection
        self.prev_tool_call_arr: list[dict] = []

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model response.

        Parses the v11+ format: [TOOL_CALLS]name{args}[TOOL_CALLS]name{args}...
        """
        # Fast path: "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"b "**********"o "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********"_ "**********"o "**********"u "**********"t "**********"p "**********"u "**********"t "**********": "**********"
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            # Get content before tool calls
            content_str = "**********"
            content: str | None = content_str if content_str.strip() else None

            # Parse tool calls from each segment after [TOOL_CALLS]
            tool_calls: list[MistralToolCall] = []
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"s "**********"e "**********"g "**********"m "**********"e "**********"n "**********"t "**********"  "**********"i "**********"n "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********"_ "**********"o "**********"u "**********"t "**********"p "**********"u "**********"t "**********". "**********"s "**********"p "**********"l "**********"i "**********"t "**********"( "**********"s "**********"e "**********"l "**********"f "**********". "**********"b "**********"o "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
                if not segment.strip():
                    continue

                matches = self.fn_name_regex.findall(segment)
                for match in matches:
                    fn_name = match[0]
                    fn_args = _escape_json_control_chars(match[1])
                    tool_calls.append(
                        MistralToolCall(
                            type="function",
                            function=FunctionCall(name=fn_name, arguments=fn_args),
                        )
                    )

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content= "**********"
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: "**********"
        current_token_ids: "**********"
        delta_token_ids: "**********"
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Extract tool calls from streaming output using token-based parsing.

        Token IDs are atomic - they cannot be split across chunks - which
        eliminates a whole class of parsing bugs that affect text-based parsing.
        """
        # If no tool call token seen yet, emit as content
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"b "**********"o "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"s "**********": "**********"
            return DeltaMessage(content=delta_text)

        # Check if this is the first chunk containing [TOOL_CALLS]
        # If so, we may have content tokens before it in this delta
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"b "**********"o "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"p "**********"r "**********"e "**********"v "**********"i "**********"o "**********"u "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"s "**********": "**********"
            return self._stream_tool_calls_with_content(delta_token_ids)

        return self._stream_tool_calls(delta_token_ids)

    def _stream_tool_calls_with_content(
        self, delta_token_ids: "**********"
    ) -> DeltaMessage | None:
        """
        Handle the first chunk containing [TOOL_CALLS].

        Content tokens before [TOOL_CALLS] are emitted as content,
        then tool call parsing begins.
        """
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        # Find where [TOOL_CALLS] appears in this delta
        assert self.bot_token_id is not None  # Validated in __init__
        try:
            bot_idx = "**********"
        except ValueError:
            # Shouldn't happen, but handle gracefully
            return self._stream_tool_calls(delta_token_ids)

        # Decode content tokens before [TOOL_CALLS]
        content_tokens = delta_token_ids[: "**********"
        content = ""
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"c "**********"o "**********"n "**********"t "**********"e "**********"n "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
            content = "**********"
                list(content_tokens),
                special_token_policy= "**********"
            )

        # Process tool call tokens (including [TOOL_CALLS] itself)
        tool_tokens = delta_token_ids[bot_idx: "**********"
        tool_result = "**********"

        # Combine content and tool calls in response
        if content and tool_result and tool_result.tool_calls:
            return DeltaMessage(content=content, tool_calls=tool_result.tool_calls)
        elif content:
            return DeltaMessage(content=content)
        else:
            return tool_result

    def _stream_tool_calls(self, delta_token_ids: "**********":
        """
        Stream tool calls using token-based parsing.

        Detects [TOOL_CALLS] and [ARGS] tokens to identify tool call boundaries,
        then streams function names and arguments as they arrive.
        """
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        delta_tool_calls: list[DeltaToolCall] = []

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"i "**********"n "**********"  "**********"d "**********"e "**********"l "**********"t "**********"a "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"s "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"= "**********"= "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"b "**********"o "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********": "**********"
                # Starting a new tool call
                self._current_tool_index += 1
                self._current_tool_id = MistralToolCall.generate_random_id()
                self._current_tool_name = ""
                self._current_tool_args = ""
                self._brace_depth = 0
                self._streaming_state = StreamingState.PARSING_TOOL_NAME

                # Set flag for finish_reason detection
                if not self.prev_tool_call_arr:
                    self.prev_tool_call_arr = [{"arguments": {}}]

                # Initialize streamed_args_for_tool for this tool index
                while len(self.streamed_args_for_tool) <= self._current_tool_index:
                    self.streamed_args_for_tool.append("")

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"= "**********"= "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"_ "**********"a "**********"r "**********"g "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********": "**********"
                # Transition from name to arguments
                if self._streaming_state == StreamingState.PARSING_TOOL_NAME:
                    # Emit the complete function name
                    delta_tool_calls.append(
                        DeltaToolCall(
                            index=self._current_tool_index,
                            type="function",
                            id=self._current_tool_id,
                            function=DeltaFunctionCall(
                                name=self._current_tool_name.strip()
                            ).model_dump(exclude_none=True),
                        )
                    )
                    self._streaming_state = StreamingState.PARSING_TOOL_ARGS

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"= "**********"= "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"_ "**********"c "**********"a "**********"l "**********"l "**********"_ "**********"i "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********": "**********"
                # Skip call ID tokens (they come between name and [ARGS])
                # We generate our own IDs
                pass

            elif self._streaming_state == StreamingState.CONTENT:
                # Before any tool call - shouldn't happen if bot_token_id
                # is in current_token_ids, but handle gracefully
                pass

            elif self._streaming_state == StreamingState.PARSING_TOOL_NAME:
                # Accumulate name tokens
                token_str = "**********"
                    [token_id], special_token_policy= "**********"
                )
                self._current_tool_name += "**********"

            elif self._streaming_state == StreamingState.PARSING_TOOL_ARGS:
                # Stream argument tokens
                token_str = "**********"
                    [token_id], special_token_policy= "**********"
                )

                # Track brace depth for nested JSON
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"c "**********"h "**********"a "**********"r "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"s "**********"t "**********"r "**********": "**********"
                    if char == "{":
                        self._brace_depth += 1
                    elif char == "}":
                        self._brace_depth -= 1

                self._current_tool_args += "**********"

                # Update streamed_args_for_tool for vLLM's finish handling
                if self._current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self._current_tool_index] = (
                        self._current_tool_args
                    )

                # Emit arguments delta
                delta_tool_calls.append(
                    DeltaToolCall(
                        index=self._current_tool_index,
                        function= "**********"=token_str).model_dump(
                            exclude_none=True
                        ),
                    )
                )

        # Build response
        if delta_tool_calls:
            return DeltaMessage(tool_calls=delta_tool_calls)

        return None
