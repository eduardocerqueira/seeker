#date: 2024-04-26T16:50:21Z
#url: https://api.github.com/gists/5c298fea3c188d5fe474ab5dcc80610e
#owner: https://api.github.com/users/BobMerkus

def switch_roles(messages: list[dict]) -> list[dict]:
    """A function to swap the roles of the messages in a conversation between a user and an AI."""
    for message in messages:
        if message["role"] == "user":
            message["role"] = "ai"
        elif message["role"] == "ai":
            message["role"] = "user"
    return messages

def converse(model_1: BaseChatModel, model_2: BaseChatModel, messages: list[dict], n_turns: int = 1) -> list[dict]:
    """A function to simulate a conversation between two models."""
    messages = messages.copy()
    for i in range(n_turns):
        output_1 = stream_response(messages, model_1, add_message=True)
        messages = switch_roles(messages)
        output_2 = stream_response(messages, model_2, add_message=True)
        messages = switch_roles(messages)
    return messages

new_messages = converse(llama_2, llama_3, messages, 3)