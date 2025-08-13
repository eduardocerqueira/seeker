#date: 2025-08-13T16:54:19Z
#url: https://api.github.com/gists/8e7c092dcd5a0d8280018d0df2f0aa3d
#owner: https://api.github.com/users/plarson

#!/usr/bin/env python3
"""
Claude Code Hook: Xcodebuild Xcpretty Validator
================================================
This hook runs as a PreToolUse hook for the Bash tool.
It validates that xcodebuild commands are piped through xcpretty for better error visibility.

Read more about hooks here: https://docs.anthropic.com/en/docs/claude-code/hooks

Configuration for ~/.claude/claude_code_config.json:
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/xcodebuild_xcpretty_validator.py"
          }
        ]
      }
    ]
  }
}
"""

import json
import re
import sys

# pyright: basic


def _validate_xcodebuild_command(command: str) -> list[str]:
    """
    Validate that xcodebuild commands are piped through xcpretty.
    Returns a list of validation issues.
    """
    issues = []
    
    # Check if command contains xcodebuild
    if 'xcodebuild' not in command:
        return issues
    
    # Pattern to detect xcodebuild that's not followed by a pipe to xcpretty
    # This handles various cases:
    # - xcodebuild at the start of command
    # - xcodebuild after && or ; or ||
    # - xcodebuild in subshells
    xcodebuild_pattern = r'(?:^|[;&|]|\()\s*xcodebuild\b'
    
    # Check if xcpretty appears after xcodebuild in a pipe
    xcpretty_pipe_pattern = r'xcodebuild[^;&|]*\|\s*(?:[^;&|]*\|\s*)*xcpretty'
    
    if re.search(xcodebuild_pattern, command):
        # Found xcodebuild, now check if it's piped to xcpretty
        if not re.search(xcpretty_pipe_pattern, command):
            issues.append(
                "xcodebuild output should be piped through xcpretty for better error visibility. "
                "Use: xcodebuild [options] | xcpretty"
            )
            
            # Additional check for common xcpretty options that might be useful
            if 'test' in command:
                issues.append(
                    "For test commands, consider: xcodebuild test [options] | xcpretty --test"
                )
            elif 'build' in command:
                issues.append(
                    "For build commands, consider: xcodebuild build [options] | xcpretty --simple"
                )
    
    return issues


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        # Exit code 1 shows stderr to the user but not to Claude
        sys.exit(1)

    tool_name = input_data.get("tool_name", "")
    if tool_name != "Bash":
        sys.exit(0)

    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")

    if not command:
        sys.exit(0)

    issues = _validate_xcodebuild_command(command)
    if issues:
        print("❌ xcodebuild command validation failed:", file=sys.stderr)
        for message in issues:
            print(f"  • {message}", file=sys.stderr)
        print("\n💡 xcpretty makes it easier to find compilation errors and warnings!", file=sys.stderr)
        # Exit code 2 blocks tool call and shows stderr to Claude
        sys.exit(2)


if __name__ == "__main__":
    main()