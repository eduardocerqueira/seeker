#date: 2025-10-28T17:13:27Z
#url: https://api.github.com/gists/e3b07f0709bc5bd09991216bba82eeb7
#owner: https://api.github.com/users/sugam0301

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import StructuredTool
from typing import Dict, Any, Callable
import inspect

# Setup server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",
    args=["math_server.py"],
)

def create_tool_function(tool_name: str, arg_schema: Dict[str, Any]) -> Callable:
    """Factory function to create a unique function for each tool"""
    
    def tool_function(**kwargs):
        """Generic function to process based on operation type"""
        print(f"Executing {tool_name} with arguments: {kwargs}")

        print("bye bye mcp")
        
        # For now, just return a message with the operation and args
        return f"Operation {tool_name} executed with args: {kwargs}"
    
    # Set the function name dynamically
    tool_function.__name__ = f"fun_{tool_name}"
    
    return tool_function



async def main():
    # Connect to the tool server via stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize MCP session
            await session.initialize()

            # Load tools exposed by the MCP server
            tools = await load_mcp_tools(session)
            print("Available tools:", [tool.name for tool in tools])

            # print(tools)

            # Create LangChain tools with unique function instances
            langchain_tools = []
            for i, tool in enumerate(tools):
                # Create a unique function for each tool
                func_instance = create_tool_function(tool.name, tool.args_schema)
                
                # print(func_instance(a=3, b=5))  # Test the function
                # Create the LangChain tool
                langchain_tool = StructuredTool.from_function(
                    name=tool.name,
                    func=func_instance,
                    description=tool.description,
                    args_schema=tool.args_schema,
                )
                langchain_tools.append(langchain_tool)
                print(f"Created tool {tool.name} with function {func_instance.__name__} , tool schema is {tool.args_schema}")

            # Create a ReAct agent using GPT-4o-mini and the tools
            agent = create_react_agent("openai:gpt-4o-mini", langchain_tools)

            # Run a query
            agent_response = await agent.ainvoke({
                "messages": "what's (3 + 5)?"
            })

            for msg in agent_response["messages"]:
                if msg.type == "ai" and msg.content:
                    print("AI Response:", msg.content)

            # final_message = agent_response["messages"][-1]
            # print("AI Response:", final_message.content)

# Run the async main
if __name__ == "__main__":
    asyncio.run(main())