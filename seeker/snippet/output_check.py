#date: 2024-04-15T16:58:39Z
#url: https://api.github.com/gists/a5cff1c9d6eacd3e124c0d0d7116e294
#owner: https://api.github.com/users/bigsnarfdude


import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

import datasets
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import evaluate
from evaluate import logging
from datasets import load_dataset
# this section calls on the downloaded weights from OpenAI
from transformers import pipeline, set_seed




# ------------------- running harness for OpenAI-GPT2 ------------------------

# openai
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

input_test_text = "What is the answer to life, the universe, and everything?"
results = generator(input_test_text, max_length=100, num_return_sequences=5)

for row in range(len(results)):
    print(results[row]['generated_text'])
    print('---------------')

#perplexity = evaluate.load("perplexity", module_type="metric")
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< openAI above babyBelow >>>>>>>>>>>>>>>>>>>>>>>>>>>>")







# ------------------- running harness for babyGPT2 ------------------------
out_dir = 'out'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5 
max_new_tokens = "**********"
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = "**********"
seed = 1337
device = 'mps'
dtype = 'float16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'mps' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)


load_meta = False
print("No meta.pkl found, assuming GPT-2 encodings...")
enc = "**********"
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


#if start is empty then use input_text --start='some question in command lne'
start_ids = encode(input_test_text)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = "**********"=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
erature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
