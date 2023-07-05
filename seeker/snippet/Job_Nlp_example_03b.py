#date: 2023-07-05T16:36:32Z
#url: https://api.github.com/gists/efea813d861cd1f6d7e31c34c3a240f9
#owner: https://api.github.com/users/gabri-al

import requests

model_name = 'sentence-transformers/all-mpnet-base-v2'
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
hf_token = 'hf_' # Token goes here, generated from https: "**********"
headers = {"Authorization": "**********"

def embed_api(sentences_):
    input_json = {"inputs": sentences_, "options":{"wait_for_model":True}}
    response = requests.post(api_url, headers=headers, json=input_json)
    return response.json()

output_embd = embed_api(query_sentence) = embed_api(query_sentence)