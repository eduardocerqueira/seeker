#date: 2024-05-03T16:57:20Z
#url: https://api.github.com/gists/727ba3eac2b42e661f2ebabaf1256115
#owner: https://api.github.com/users/av1d

import json
import random
import re
import requests
import textwrap

# a random model is selected each time without the possibility
# of selecting the same one twice in a row (so it doesn't speak to itself)

# Terminal colors
BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'
# lighter colors
LIGHT_RED = '\033[91m'
LIGHT_GREEN = '\033[92m'
LIGHT_YELLOW = '\033[93m'
LIGHT_BLUE = '\033[94m'
LIGHT_MAGENTA = '\033[95m'
LIGHT_CYAN = '\033[96m'
LIGHT_GRAY = '\033[37m'
# reset colors
RESET = '\033[0m'

# server addresses
NPU_ADDRESS = "192.168.0.196"
NPU_PORT = "31337"
LLAMA_IP = "192.168.0.196"
LLAMA_PORT = "8080"

# servers
server1 = "npu"
server2 = "ollama"
server3 = "llamacpp"
# remembers the last server used so we don't prompt
# the same one twice in a row
last_picked = None


def npu(query: str) -> str:

    headers = {
        'Content-Type': 'application/json',
    }

    prefix = (
        "<|im_start|>system You are a helpful assistant. <|im_end|> "
        "<|im_start|>user "
    )

    postfix = (
        "<|im_end|><|im_start|>assistant "
    )

    json_data = {
        'PROMPT_TEXT_PREFIX': prefix,
        'input_str': str(query) + ' ',
        'PROMPT_TEXT_POSTFIX': postfix,
    }

    response = requests.post(
        f"http://{NPU_ADDRESS}:{NPU_PORT}",
        headers=headers,
        json=json_data
    )

    response = response.json()

    answer = response['content']

    return answer

def llamacpp(query: str) -> str:

    prompt_text = (
        f"This is a conversation between User and Llama, a friendly "
        f"chatbot. Llama is helpful, kind, honest, good at writing, and "
        f"never fails to answer any requests immediately and with "
        f"precision.\n\n"
        f"User: {query} "
        f"\nLlama:"
    )

    prompt_text = (
        f"This is a conversation between User and Llama, an analyitically-minded "
        f"thinker who deconstructs everything everyone says and analyzes it for logical fallacy. "
        f"Lama refutes and contests anything which doesn't have scientific foundations. "
        f"User: {query} "
        f"\nLlama:"
    )

    headers = {
        'Accept': 'text/event-stream',
        'Accept-Language': 'en-US,en;q=0.9,it;q=0.8',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
    }

    json_data = {
        'stream': True,
        'n_predict': 400,
        'temperature': 0,
        'stop': [
            '</s>',
            'Llama:',
            'User:',
        ],
        'repeat_last_n': 256,
        'repeat_penalty': 2,
        'top_k': 40,
        'top_p': 0.95,
        'min_p': 0,
        'tfs_z': 1,
        'typical_p': 1,
        'presence_penalty': 0,
        'frequency_penalty': 0,
        'mirostat': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'grammar': '',
        'n_probs': 0,
        'min_keep': 0,
        'image_data': [],
        'cache_prompt': True,
        'api_key': '',
        'slot_id': -1,
        'prompt': prompt_text,
    }

    response = requests.post(
        'http://192.168.0.196:8080/completion',
        headers=headers,
        json=json_data,
        verify=False
    )

    data = response.content.decode('utf-8')
    lines = data.split('\n')
    content = ''

    for line in lines:
        if line.startswith('data: '):
            json_data = json.loads(line[6:])
            content += json_data['content']

    return content

def ollama(query: str) -> str:
    headers = {
        'Content-Type': 'application/json',
    }

    data = {
        "model": "tinyllama:latest",
        "prompt": query
    }

    response = requests.post('http://localhost:11434/api/generate', headers=headers, json=data)

    json_response = response.content.decode('utf-8')

    lines = json_response.strip().split('\n')

    concatenated_response = ''

    for line in lines:
        try:
            data = json.loads(line)
            concatenated_response += data['response']
        except json.JSONDecodeError:
            pass

    return concatenated_response

def trim_text(text: str) -> str:
    # split sentences at .!? into list
    result = re.split(r'(?<=[.!?]) +', text)
    # keep only the first N
    result = result[:10]
    # convert back to string
    result = ''.join(result)
    return result

def select_server() -> str:

    global last_picked

    while True:
        if last_picked == server1:
            choices = [server2, server3]
        elif last_picked == server2:
            choices = [server1, server3]
        elif last_picked == server3:
            choices = [server1, server2]
        else:
            choices = [server1, server2, server3]

        choice = random.choice(choices)

        last_picked = choice

        print(f"{GREEN}[chose server: {choice}]{RESET}")

        return choice


start = "What is the weirdest thing you know?"
print(f"Starting bias seed: {start}\n")

for i in range(10):

    server = select_server()

    if server == 'ollama':
        start = trim_text(ollama(start))
        start = textwrap.fill(start, width=80)
        print("[TinyLlama]: " + LIGHT_GRAY + str(start) + RESET + "\n")
    elif server == 'npu':
        start = trim_text(npu(start))
        start = textwrap.fill(start, width=80)
        print("[Qwen]: " + LIGHT_CYAN + str(start) + RESET + "\n")
    elif server == 'llamacpp':
        start = trim_text(llamacpp(start))
        start = textwrap.fill(start, width=80)
        print("[Zephyr]: " + LIGHT_YELLOW + str(start) + RESET + "\n")
