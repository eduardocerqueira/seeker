#date: 2025-03-26T17:03:13Z
#url: https://api.github.com/gists/b9611e5924a047dac7eb17a9b55a7e4e
#owner: https://api.github.com/users/megha-shroff

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    return a - b
