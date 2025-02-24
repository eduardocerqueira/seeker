#date: 2025-02-24T16:58:14Z
#url: https://api.github.com/gists/d6805d2448d1cce51230876992a01e7e
#owner: https://api.github.com/users/danyaljj

import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cpu"
tokenizer = "**********"
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

dur = {}
for use_KV in (True, False):
    dur[use_KV]=[]
    for _ in range(3):  # measuring 3 generations
        start = time.time()
        response = "**********"="pt").to(device), 
                                  use_cache= "**********"=100)
        dur[use_KV].append(time.time() - start)

for use_KV in (True, False):
    setting = 'with' if use_KV else 'without'
    print(f"{setting} KV cache: {round(np.mean(dur[use_KV]),3)} ± {round(np.std(dur[use_KV]),3)} seconds")
print(f"Ratio of Without/With: x {round(np.mean(dur[False])/np.mean(dur[True]),3)} speed-up\n")

print("*****\n"+tokenizer.decode(response[0], skip_special_tokens= "**********",3)} speed-up\n")

print("*****\n"+tokenizer.decode(response[0], skip_special_tokens=True)+"\n*****")