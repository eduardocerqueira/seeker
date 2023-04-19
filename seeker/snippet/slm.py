#date: 2023-04-19T17:03:20Z
#url: https://api.github.com/gists/3395f866e0c1a72f134ba4885d8c1bf7
#owner: https://api.github.com/users/cmsj

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from IPython.display import Markdown, display
def hr(): display(Markdown('---'))
def cprint(msg: str, color: str = "blue", **kwargs) -> str:
    if color == "blue": print("\033[34m" + msg + "\033[0m", **kwargs)
    elif color == "red": print("\033[31m" + msg + "\033[0m", **kwargs)
    elif color == "green": print("\033[32m" + msg + "\033[0m", **kwargs)
    elif color == "yellow": print("\033[33m" + msg + "\033[0m", **kwargs)
    elif color == "purple": print("\033[35m" + msg + "\033[0m", **kwargs)
    elif color == "cyan": print("\033[36m" + msg + "\033[0m", **kwargs)
    else: raise ValueError(f"Invalid info color: `{color}`")

# Choose model name
model_name = "stabilityai/stablelm-base-alpha-7b" #@param ["stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-7b", "stabilityai/stablelm-base-alpha-3b", "stabilityai/stablelm-tuned-alpha-3b"]

cprint(f"Using `{model_name}`", color="blue")
if torch.cuda.is_available():
    cprint("CUDA is available", color="green")
else:
    cprint("CUDA is not available", color="red")
    import sys
    sys.exit(0)

# Select "big model inference" parameters
torch_dtype = "float16" #@param ["float16", "bfloat16", "float"]
load_in_8bit = False #@param {type:"boolean"}
device_map = "auto"

cprint(f"Loading with: `{torch_dtype=}, {load_in_8bit=}, {device_map=}`")

tokenizer = "**********"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=getattr(torch, torch_dtype),
    load_in_8bit=load_in_8bit,
    device_map=device_map,
    offload_folder="./offload",
)

cprint(f"Loaded model: `{model_name}` ({model.device})", color="green")

prompt = "Define relativity" #@param {type:"string"}

# Sampling args
max_new_tokens = 64 #@param {type: "**********":32.0, max:3072.0, step:32}
temperature = 0.5 #@param {type:"slider", min:0.0, max:1.25, step:0.05}
top_k = 0 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
top_p = 0.9 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
do_sample = True #@param {type:"boolean"}

cprint(f"Sampling with: "**********"
hr()

# Create `generate` inputs
inputs = "**********"="pt")
inputs.to(model.device)

# Generate
tokens = "**********"
  **inputs,
  max_new_tokens= "**********"
  temperature=temperature,
  top_k=top_k,
  top_p=top_p,
  do_sample=do_sample,
  pad_token_id= "**********"
)

# Extract out only the completion tokens
completion_tokens = tokens[0 "**********"[inputs['input_ids' "**********".size(1): "**********"
completion = "**********"=True)

# Display
print(prompt, end="")
cprint(completion, color="green")
ompletion_tokens = tokens[0][inputs['input_ids'].size(1):]
completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

# Display
print(prompt, end="")
cprint(completion, color="green")
