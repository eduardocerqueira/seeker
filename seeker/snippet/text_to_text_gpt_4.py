#date: 2024-01-03T16:51:37Z
#url: https://api.github.com/gists/4fef522df37ce6c5b3a0f4fc734f3e91
#owner: https://api.github.com/users/smitAtSG

import sys
from openai import OpenAI
import tiktoken

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")


 "**********"d "**********"e "**********"f "**********"  "**********"n "**********"u "**********"m "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"_ "**********"f "**********"r "**********"o "**********"m "**********"_ "**********"m "**********"e "**********"s "**********"s "**********"a "**********"g "**********"e "**********"s "**********"( "**********"m "**********"e "**********"s "**********"s "**********"a "**********"g "**********"e "**********"s "**********", "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********"= "**********"" "**********"g "**********"p "**********"t "**********"- "**********"3 "**********". "**********"5 "**********"- "**********"t "**********"u "**********"r "**********"b "**********"o "**********"- "**********"0 "**********"6 "**********"1 "**********"3 "**********"" "**********") "**********": "**********"
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = "**********"
    except KeyError:
        # print("Warning: model not found. Using cl100k_base encoding.")
        encoding = "**********"
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = "**********"
        tokens_per_name = "**********"
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = "**********"
        tokens_per_name = "**********"
    elif "gpt-3.5-turbo" in model:
        # print("Warning: "**********"
        return num_tokens_from_messages(messages, model= "**********"
    elif "gpt-4" in model:
        # print("Warning: "**********"
        return num_tokens_from_messages(messages, model= "**********"
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https: "**********"
        )
    num_tokens = "**********"
    for message in messages:
        num_tokens += "**********"
        for key, value in message.items():
            num_tokens += "**********"
            if key == "name":
                num_tokens += "**********"
    num_tokens += "**********"
    return num_tokens


personality_template = "you as dumbledore from harry potter, talk about harry potter only"
gptmodel =   "gpt-3.5-turbo" #"gpt-4"
system_role = {"role": "system", "content": personality_template}

system_role_tokens = "**********"
    [system_role], gptmodel
)
print(f"system_role_tokens: "**********"

"""
# set this according to the model max context limit or bit less
# keeping the MAX_TOKENS value smaller for the demonstration purpose
"""
MAX_TOKENS = "**********"
max_tokens = "**********"


max_tokens -= "**********"
print(f" max_tokens -= "**********",  max_tokens: "**********"

conversation_history = []
conversation_history.insert(0, system_role)

try:
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        conversation_history.append({"role": "user", "content": user_input})

        while True:
            total_tokens = "**********"
                conversation_history, gptmodel
            )
            # print(f"conversation_history total_tokens: "**********"
            """
            logic for placing 250 static
            after token limit exceed it removes the earlier conversation
            so based on the early message tokens we get our new "max_tokens" 
            which determines the ans limit of model respnse.
            it was usually very less tokens to get resonse after max token scenario,
            so reserving 250 tokens for AI answer token limit
            """
            if total_tokens <= max_tokens - 250 : "**********"
                print(f"total_tokens{total_tokens} <= "**********"
                break
            else:
                conversation_history.pop(0)  # Remove the earliest message
                print("Remove the earliest message")

        # adding personality prompt at first place of the list to persists the personality
        # tokens were reserved earlier so not gonna be the issue
        conversation_history.insert(0, system_role)
        # print(f"processed conversation_history: {conversation_history} ") # you can observe the conversation_history by uncomment this

        """
        code for estimating token usage,
        this will help for further design making and troubleshooting,
        when we decide to use diffrent model
        """

        estimated_token_usage = "**********"
            conversation_history, gptmodel
        )
        
        print(
            f"{estimated_token_usage} prompt tokens counted by num_tokens_from_messages(). included system"
        )

        response = client.chat.completions.create(
            model=gptmodel,
            messages=conversation_history,
            max_tokens= "**********"
            stop=None,
        )

        assistant_response = response.choices[0].message.content
        print(f"Dumbledore: {assistant_response}")

        conversation_history.append({"role": "assistant", "content": assistant_response})
except KeyboardInterrupt:
    print("by")
")
