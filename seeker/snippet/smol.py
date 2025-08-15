#date: 2025-08-15T17:10:23Z
#url: https://api.github.com/gists/a1052e6db7a97c50bea9ef2f4da5e4bd
#owner: https://api.github.com/users/ai-christianson

# example tiny local agent by A.I. Christianson, founder of gobii.ai, builder of ra-aid.ai
#
# to run: uv run --with 'smolagents[mlx-lm]' --with ddgs smol.py 'how much free disk space do I have?'

from smolagents import CodeAgent, MLXModel, tool
from subprocess import run
import sys

@tool
def write_file(path: str, content: str) -> str:
    """Write text.
    Args:
      path (str): File path.
      content (str): Text to write.
    Returns:
      str: Status.
    """
    try:
        open(path, "w", encoding="utf-8").write(content)
        return f"saved:{path}"
    except Exception as e:
        return f"error:{e}"

@tool
def sh(cmd: str) -> str:
    """Run a shell command.
    Args:
      cmd (str): Command to execute.
    Returns:
      str: stdout+stderr.
    """
    try:
        r = run(cmd, shell=True, capture_output=True, text=True)
        return r.stdout + r.stderr
    except Exception as e:
        return f"error:{e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python agent.py 'your prompt'"); sys.exit(1)
    common = "use cat/head to read files, use rg to search, use ls and standard shell commands to explore."
    agent = CodeAgent(
        model= "**********"="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2", max_tokens=8192, trust_remote_code=True),
        tools=[write_file, sh],
        add_base_tools=True,
    )
    print(agent.run(" ".join(sys.argv[1:]) + " " + common))

))

