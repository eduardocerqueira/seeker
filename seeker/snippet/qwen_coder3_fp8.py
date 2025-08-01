#date: 2025-07-31T16:58:49Z
#url: https://api.github.com/gists/a28b6753904d030298609b5b29bee84a
#owner: https://api.github.com/users/bigsnarfdude

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

# load the tokenizer and the model
tokenizer = "**********"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Write a quick sort algorithm."
messages = [
    {"role": "user", "content": prompt}
]
text = "**********"
    messages,
    tokenize= "**********"
    add_generation_prompt=True,
)
model_inputs = "**********"="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens= "**********"
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = "**********"=True)

print("content:", content)
nt = "**********"=True)

print("content:", content)
s=True)

print("content:", content)
