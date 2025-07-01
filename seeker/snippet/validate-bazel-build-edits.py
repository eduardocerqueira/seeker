#date: 2025-07-01T16:44:13Z
#url: https://api.github.com/gists/c9197a645ed941457c6a59c5a35a3d8e
#owner: https://api.github.com/users/ewhauser

#!/usr/bin/env python3
import json
import sys
import os

try:
    input_data = json.load(sys.stdin)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
    sys.exit(1)

tool_name = input_data.get("tool_name", "")
tool_input = input_data.get("tool_input", {})

# Only process Edit, MultiEdit, and Write tool calls
if tool_name not in ["Edit", "MultiEdit", "Write"]:
    sys.exit(0)

file_path = tool_input.get("file_path", "")

# Check if this is a BUILD.bazel file
if not file_path.endswith("BUILD.bazel"):
    sys.exit(0)

# Check if the operation contains a "# keep" directive
content_to_check = ""

if tool_name == "Edit":
    # For Edit tool, check new_string
    content_to_check = tool_input.get("new_string", "")
elif tool_name == "MultiEdit":
    # For MultiEdit tool, check all new_strings in edits
    edits = tool_input.get("edits", [])
    for edit in edits:
        content_to_check += edit.get("new_string", "") + "\n"
elif tool_name == "Write":
    # For Write tool, check content
    content_to_check = tool_input.get("content", "")

# Allow if the content being added contains "# keep"
if "# keep" in content_to_check:
    sys.exit(0)

# Block the edit and provide guidance
error_message = """Please use gazelle to generate and update any BUILD.bazel files.

Run: bazel run //:gazelle

If gazelle cannot generate the change you need to make, then add a "# keep" directive at the end of one of the lines in the BUILD.bazel file to indicate manual editing is required."""

print(error_message, file=sys.stderr)
sys.exit(2)
