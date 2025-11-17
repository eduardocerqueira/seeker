#date: 2025-11-17T17:11:37Z
#url: https://api.github.com/gists/3aee40d00a7b38c6c197bff9b2eed1a6
#owner: https://api.github.com/users/sydney-runkle

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware


agent = create_agent(
    # simulate model failure w/ a typo in the model name (404)
    model="anthropic:claude-sonnet-with-typo",
    middleware=[
        ModelFallbackMiddleware("openai:gpt-4o-with-typo", "openai:gpt-4o-mini")
    ],
)