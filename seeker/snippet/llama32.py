#date: 2024-09-27T16:51:01Z
#url: https://api.github.com/gists/2dcf77db45fa6e6e77767c8244d1f150
#owner: https://api.github.com/users/tcapelle

import os, openai, weave

MODEL = "Llama-3.2-90B-Vision-Instruct"

weave.init("EU_HAS_LLAMA_90B")

image_url = "https://limaspanishhouse.com/wp-content/uploads/2021/02/peruvian-llama-2-1536x1346.jpg"
llama_client = openai.OpenAI(
    base_url="http://195.242.25.198:8032/v1",
    api_key=os.environ.get("WANDB_API_KEY")
)
@weave.op
def call_llama32(image_url, prompt="What's on the image?"):
    messages = [{
        "role": "user",
        "content": [{"type": "text","text": prompt},
                    {"type": "image_url","image_url": {"url": image_url}}]}]
    response = llama_client.chat.completions.create(
        messages=messages,
        model=MODEL)
    return response.choices[0].message.content

print("="*100)    
print(call_llama32(image_url))


