#date: 2025-08-13T17:08:17Z
#url: https://api.github.com/gists/b0841cd0e500152cb7333fe968582812
#owner: https://api.github.com/users/plarson

#!/usr/bin/env python3
"""
Claude Code Hook: UV Command Modernizer
=========================================
This hook runs as a PreToolUse hook for the Bash tool.
It detects legacy uv pip commands and suggests modern uv workflow equivalents.

The modern uv workflow uses:
- uv sync: Install all dependencies from pyproject.toml
- uv add <package>: Add a new dependency
- uv remove <package>: Remove a dependency
- uv lock: Update the lock file
- uv run: Run commands in the virtual environment

Configuration for claude_config.json:
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/uv_command_modernizer.py"
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

# Map of uv pip patterns to modern equivalents
UV_MODERNIZATION_RULES = [
    (
        r"^uv\s+pip\s+install\s+-r\s+requirements\.txt",
        "Use 'uv sync' to install all dependencies from pyproject.toml (migrate requirements.txt to pyproject.toml first)"
    ),
    (
        r"^uv\s+pip\s+install\s+(?!.*-[er])\s*([^-].*?)(?:\s|$)",
        "Use 'uv add {packages}' to add dependencies to pyproject.toml and install them"
    ),
    (
        r"^uv\s+pip\s+install\s+-e\s+\.",
        "Use 'uv sync' for editable installs (uv automatically installs the project in editable mode)"
    ),
    (
        r"^uv\s+pip\s+uninstall\s+(.*?)(?:\s|$)",
        "Use 'uv remove {packages}' to remove dependencies from pyproject.toml"
    ),
    (
        r"^uv\s+pip\s+freeze",
        "Use 'uv lock' to create/update uv.lock file, or 'uv tree' to see installed packages"
    ),
    (
        r"^uv\s+pip\s+list",
        "Use 'uv tree' to see the dependency tree of installed packages"
    ),
    (
        r"^uv\s+pip\s+show\s+(.*?)(?:\s|$)",
        "Use 'uv tree' or check pyproject.toml/uv.lock for package information"
    ),
    (
        r"^uv\s+pip\s+compile",
        "Use 'uv lock' to generate a lock file with resolved dependencies"
    ),
    (
        r"^uv\s+pip\s+sync",
        "Just use 'uv sync' - the modern command doesn't require the 'pip' subcommand"
    ),
]

def extract_packages(command: str, pattern: str) -> str:
    """Extract package names from the command for substitution."""
    match = re.search(pattern, command)
    if match and len(match.groups()) > 0:
        packages = match.group(1).strip()
        # Clean up common flags that might be captured
        packages = re.sub(r'-[a-zA-Z]\s+\S+\s*', '', packages)
        packages = re.sub(r'--\S+\s*', '', packages)
        return packages.strip()
    return ""

def get_modern_suggestion(command: str) -> tuple[bool, str]:
    """
    Check if command uses legacy uv pip and return modern equivalent.
    Returns (is_legacy, suggestion_message)
    """
    command = command.strip()
    
    for pattern, suggestion_template in UV_MODERNIZATION_RULES:
        if re.search(pattern, command, re.IGNORECASE):
            # Handle package extraction for install/uninstall/show commands
            if '{packages}' in suggestion_template:
                packages = extract_packages(command, pattern)
                if packages:
                    suggestion = suggestion_template.format(packages=packages)
                else:
                    suggestion = suggestion_template.replace('{packages}', '<packages>')
            else:
                suggestion = suggestion_template
            
            return True, suggestion
    
    return False, ""

def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    tool_name = input_data.get("tool_name", "")
    if tool_name != "Bash":
        sys.exit(0)

    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")

    if not command:
        sys.exit(0)

    # Check if this is a legacy uv pip command
    is_legacy, suggestion = get_modern_suggestion(command)
    
    if is_legacy:
        print("🚀 Modern uv workflow detected!", file=sys.stderr)
        print(f"• {suggestion}", file=sys.stderr)
        print("", file=sys.stderr)
        print("The modern uv workflow uses pyproject.toml instead of requirements.txt:", file=sys.stderr)
        print("• uv init - Initialize a new project", file=sys.stderr)
        print("• uv add <package> - Add dependencies", file=sys.stderr)
        print("• uv remove <package> - Remove dependencies", file=sys.stderr)
        print("• uv sync - Install all dependencies", file=sys.stderr)
        print("• uv lock - Update the lock file", file=sys.stderr)
        print("• uv run <command> - Run commands in the venv", file=sys.stderr)
        print("• uv tree - Show dependency tree", file=sys.stderr)
        
        # Exit code 2 blocks the tool call and shows stderr to Claude
        sys.exit(2)

if __name__ == "__main__":
    main()