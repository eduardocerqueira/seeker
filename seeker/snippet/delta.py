#date: 2025-05-02T16:50:42Z
#url: https://api.github.com/gists/76e6ddad16fb698bd9f39ee06a5bacaa
#owner: https://api.github.com/users/data2json

import torch
from transformers import AutoModelForCausalLM

# Load models
llama3_base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
llama3_inst = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
llama31_base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# Calculate Δθ = θpost - θbase
delta_params = {}
for name, param in llama3_base.named_parameters():
delta_params[name] = llama3_inst.get_parameter(name) - param
# Create Param∆ model: θParam∆ = θ'base + Δθ
param_delta_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
for name, param in param_delta_model.named_parameters():
if name in delta_params:
param.data += delta_params[name]

# Save the resulting model
param_delta_model.save_pretrained("llama31-with-llama3-inst-capabilities")