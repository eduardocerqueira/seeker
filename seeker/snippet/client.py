#date: 2025-02-24T16:53:13Z
#url: https://api.github.com/gists/0fd79b8db128ae8887b4d5c9eede21eb
#owner: https://api.github.com/users/thoraxe

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import os
import subprocess

import fire

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.client_tool import client_tool
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig

pre_path = "/home/thoraxe/bin/"


@client_tool
async def get_object_namespace_list(kind: str, namespace: str) -> str:
    """Get the list of all objects in a namespace

    :param kind: the type of object
    :param namespace: the name of the namespace
    :returns: a plaintext list of the kind object in the namespace
    """
    output = subprocess.run(
        [pre_path + "oc", "get", kind, "-n", namespace, "-o", "name"],
        capture_output=True,
        timeout=2,
    )
    return output.stdout


async def run_main(host: str, port: int, user_query: str, disable_safety: bool = False):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print("No available shields. Disable safety.")
    else:
        print(f"Available shields found: {available_shields}")

    client_tools = [get_object_namespace_list]

    agent_config = AgentConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        instructions="""You are a helpful assistant with access to the following
function calls. Your task is to produce a list of function calls
necessary to generate response to the user utterance. Use the following
function calls as required.""",
        toolgroups=[],
        client_tools=[
            client_tool.get_tool_definition() for client_tool in client_tools
        ],
        tool_choice="auto",
        tool_prompt_format="python_list",
        enable_session_persistence=False,
    )

    agent = Agent(client, agent_config, client_tools)
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": user_query,
            }
        ],
        session_id=session_id,
    )

    for log in EventLogger().log(response):
        log.print()

def main(host: str, port: int, user_query: str):
    asyncio.run(run_main(host, port, user_query))


if __name__ == "__main__":
    fire.Fire(main)
