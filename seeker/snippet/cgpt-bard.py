#date: 2023-04-26T17:01:04Z
#url: https://api.github.com/gists/300e4f68b3ac6581e69e77fe42d812a1
#owner: https://api.github.com/users/teshanshanuka

#!/usr/bin/python3.8
# Author: Teshan Liyanage <teshanuka@gmail.com>

import readline
import openai
from Bard import Chatbot  # from https://github.com/acheong08/Bard
import time

# ANSI colors
c_b, c_g, c_gr, c_nc = "\033[94m", "\033[32m", "\033[30m", "\033[0m"


# ChatGPT
# api key from https://platform.openai.com/account/api-keys
openai.api_key = "X"
cgpt_default_role = "You are a helpful assistant"
cgpt_chat_log = [{"role": "system", "content": cgpt_default_role}]
cgpt_completion = openai.ChatCompletion()
cgpt_model = "gpt-3.5-turbo"


# Bard
# token from browser: "**********"
bard_token = "**********"
bard_completion = "**********"


def cgpt(q):
    cgpt_chat_log.append({"role": "user", "content": q})
    resp = cgpt_completion.create(model=cgpt_model, messages=cgpt_chat_log)
    c = resp.choices[0]

    cgpt_chat_log.append({"role": "assistant", "content": c.message.content})
    print(c_b + "\nC:" + c.message.content + c_nc)
    return c.message.content


def bard(q):
    resp = bard_completion.ask(q)
    print(c_g + "\nB:" + resp['content'] + c_nc)
    return resp['content']

if __name__ == "__main__":
    first_msg = input("Start conversation msg > ")
    if (m := input("Who starts? [C|b]")) == "b":
        resp = bard(first_msg)
    else:
        resp = first_msg

    for _ in range(10):
        resp = cgpt(resp)
        resp = bard(resp)
        time.sleep(20)  # openai has a tokens/min limit
        time.sleep(20)  # openai has a tokens/min limit
