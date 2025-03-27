#date: 2025-03-27T17:11:34Z
#url: https://api.github.com/gists/1cc739c2d08c30fde0154a87be737d3f
#owner: https://api.github.com/users/seratch

#
# OpenAI Agents SDK MCP server example
#
# How to run this app:
# $ pip install -U openai-agents
# $ python app.py
#
# See also:
#  - https://openai.github.io/openai-agents-python/mcp/
#  - https://github.com/openai/openai-agents-python/tree/main/examples/mcp
#  - https://github.com/modelcontextprotocol/servers/tree/main/src/slack

"""
The App Manifest for the MCP Connector:
Go to https://api.slack.com/apps and then create a new app using the following YAML:

display_information:
  name: Test MCP Connector
features:
  bot_user:
    display_name: Test MCP Connector
    always_online: false
oauth_config:
  scopes:
    user:
      - chat:write
      - channels:history
      - channels:read
      - groups:history
      - groups:read
    bot:
      - channels:history
      - channels:read
      - chat:write
      - reactions:write
      - users:read
settings:
  org_deploy_enabled: false
  socket_mode_enabled: false
  token_rotation_enabled: "**********"
"""

import asyncio

import os
import shutil
import logging

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServerStdio


logging.basicConfig(level=logging.WARNING)

# If you want to see more logs, enable the following lines:
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("openai.agents").setLevel(logging.DEBUG)


async def main():
    command = "npx -y @modelcontextprotocol/server-slack"

    # Two environment variables must be set:
    # export SLACK_BOT_TOKEN= "**********"
    # epxort SLACK_TEAM_ID=T....
    env = {
        "SLACK_BOT_TOKEN": "**********"
        "SLACK_TEAM_ID": os.environ["SLACK_TEAM_ID"],
    }
    params = {
        "command": command.split(" ")[0],
        "args": command.split(" ")[1:],
        "env": env,
    }
    async with MCPServerStdio(
        name="Filesystem Server via npx", params=params
    ) as slack_server:
        trace_id = gen_trace_id()
        with trace(workflow_name="MCP Slack Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/{trace_id}\n")
            agent = Agent(
                name="MCP Slack Agent",
                instructions="Use the tools to access Slack workspaces.",
                mcp_servers=[slack_server],
            )
            print("Accessing Slack using @modelcontextprotocol/server-slack ...")

            # Example 1: "**********":
            # Invite "Test MCP Connector" to the following channel:
            prompt = "Tell me the message summary within the last 24 hours in #team-developer-experience (C08KKPMSKGD)"
            result = await Runner.run(starting_agent=agent, input=prompt)
            print(result.final_output)

            # Example 2: "**********":
            # prompt = "Tell me the message summary within the last 24 hours in #team-developer-experience (C08KKPMSKGD), and then say good morning in the same channel on behalf of me."
            # result = await Runner.run(starting_agent=agent, input=prompt)
            # print(result.final_output)


if __name__ == "__main__":
    if not shutil.which("npx"):
        error = "npx is not installed. Please install it with `npm install -g npx`."
        raise RuntimeError(error)

    asyncio.run(main())
     raise RuntimeError(error)

    asyncio.run(main())
