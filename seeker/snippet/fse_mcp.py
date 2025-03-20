#date: 2025-03-20T17:00:15Z
#url: https://api.github.com/gists/acf752ed5d84fc0793126f637969e5f3
#owner: https://api.github.com/users/minpeter

import asyncio
import json
import os
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.friendli = OpenAI(
            base_url="https://api.friendli.ai/serverless/v1",
            api_key= "**********"
        )

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [{"role": "user", "content": query}]

        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]

        # Initial Claude API call
        response = self.friendli.chat.completions.create(
            model="meta-llama-3.3-70b-instruct",
            max_tokens= "**********"
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
        )

        # Process response and handle tool calls
        final_text = []
        assistant_message = response.choices[0].message

        # Add assistant's message to the conversation history
        messages.append(
            {"role": "assistant", "content": assistant_message.content or ""}
        )

        # Check if there are tool calls
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                # Extract tool call information
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments

                # Parse JSON string into Python dictionary
                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError as e:
                    final_text.append(f"[Error parsing tool arguments: {str(e)}]")
                    continue

                # Execute tool call with parsed dictionary
                result = await self.session.call_tool(tool_name, tool_args)

                print(result)

                final_text.append(
                    f"[Calling tool {tool_name} with args {tool_args_str}]"
                )

                # Add tool result to conversation history
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result.content,
                    }
                )

            # Get next response from the model with tool results
            response = self.friendli.chat.completions.create(
                model="meta-llama-3.3-70b-instruct",
                max_tokens= "**********"
                messages=messages,
            )

            # Add final assistant response to output
            assistant_response = response.choices[0].message.content
            final_text.append(assistant_response)
            messages.append({"role": "assistant", "content": assistant_response})
        else:
            # If no tool calls, just add the assistant's message
            final_text.append(assistant_message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():

    MCP_SERVER_URL = "https://w-mcp.minpeter.xyz/sse"
    client = MCPClient()
    try:
        await client.connect_to_sse_server(MCP_SERVER_URL)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
())
