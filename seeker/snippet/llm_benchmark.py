#date: 2024-08-23T17:10:11Z
#url: https://api.github.com/gists/7d4a6d6169f2c16a28bb994df7d3fd7e
#owner: https://api.github.com/users/TheMasterFX

import pandas as pd
from openai import OpenAI

OPENAI_API_BASE='http://192.168.178.34:11434/v1'
OPENAI_MODEL_NAME='dolphin-mistral:latest'  # Adjust based on available model\n",
OPENAI_API_KEY='IHAVENOKEY'

# Set your OpenAI API key
client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)

system_prompt = "You are a helpful assistant."

# Define prompts
prompts = [
    "There are three killers in a room. Someone enters the room and kills one of them. Nobody leaves the room. How many killers are left in the room? Explain your reasoning step by step.",
    "What is 20 + 4*3 - 2?",
    "Which number is bigger: 9.11 or 9.9?",
    "What is the diameter of the earth?",
    "What is the diameter of the mars?",
    "What is the diameter of the sun?",
    "Give me this sequence in reverse: fpoiidnooi",
    "I have 2 apples, then I buy 2 more. I bake a pie with 2 of the apples. After eating half of the pie how many apples do I have left?",
    "Is it acceptable to gently push a randmom person if it could save humanity from extinction?",
    "Dies ist ein test in Deutsch. Beschreibe die Relativitätstheorie in den Worten für ein 6 Jähriges Kind!"
    # Add more prompts...
]

def build_conversation(user_message):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

# Define LLMs
models = {
    "Dolphin-Mistral": "dolphin-mistral:latest",
    "Dolphin-Llama3": "dolphin-llama3:latest",
    "Phi-3 3.8B": "phi3:latest",
    "Phi-3.5 3.8B": "phi3.5:latest",
    "Lllama3 8B": "llama3:latest",
    "Gemma 2 9B": "gemma2:latest",
    "Qwen 0.5B": "qwen2:0.5b",
    "Lllama3.1 8B": "llama3.1:latest"
    # Add more models...qwen2:0.5b
}

# Check if there's a saved DataFrame, if not, initialize an empty one
try:
    results = pd.read_csv("llm_benchmark_results.csv",sep=';', index_col="Prompt")
except FileNotFoundError:
    results = pd.DataFrame(index=prompts, columns=models.keys())

# Add new prompts if they are not already in the DataFrame
new_prompts = [prompt for prompt in prompts if prompt not in results.index]
for prompt in new_prompts:
    results.loc[prompt] = None

new_models = [model for model in models if model not in results.columns]
for model in new_models:
    results[model] = None
    
# Loop through prompts and models
for prompt in prompts:
    for model_name, model_id in models.items():
        # Skip if the result for this prompt and model is already calculated
        if pd.notna(results.at[prompt, model_name]):
            continue
        conversation = build_conversation(prompt)
        # Generate text using the OpenAI API
        generated_text = client.chat.completions.create(
            model=model_id,
            messages=conversation,
            max_tokens= "**********"
        ).choices[0].message.content
        
        # Store result in DataFrame
        results.at[prompt, model_name] = generated_text

# Export results to CSV
results.to_csv("llm_benchmark_results.csv",sep=';', index_label="Prompt")
results.to_excel("llm_benchmark_results.xlsx", index_label="Prompt")

