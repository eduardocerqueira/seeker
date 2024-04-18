#date: 2024-04-18T16:51:27Z
#url: https://api.github.com/gists/59b18273a61b22f009e946e0d07b4c1f
#owner: https://api.github.com/users/bigsnarfdude

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = "**********"
        messages, 
        tokenize= "**********"
        add_generation_prompt=True
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens= "**********"
    eos_token_id= "**********"
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
prompt):])
