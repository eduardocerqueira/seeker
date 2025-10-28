#date: 2025-10-28T17:13:27Z
#url: https://api.github.com/gists/e3b07f0709bc5bd09991216bba82eeb7
#owner: https://api.github.com/users/sugam0301

# math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print(f"Adding {a} and {b}")
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")