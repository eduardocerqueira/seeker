#date: 2025-03-26T17:08:26Z
#url: https://api.github.com/gists/63855c6e94e2c3cc2f25aa385a099b57
#owner: https://api.github.com/users/megha-shroff

@mcp.prompt("operation-decider")
def operation_decider_prompt(user_query: str) -> list[base.Message]:
    return [
        base.UserMessage(f"""Strictly extract numbers and operation from: {user_query}. Reply **ONLY in JSON string** like this: {{"a": a, "b": b, "operation": "add / subtract"}}"""),
    ]

if __name__ == "__main__":
    print("MCP server is running using stdio transport ...")
    mcp.run(transport="stdio")
