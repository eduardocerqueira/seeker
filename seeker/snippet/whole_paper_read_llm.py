#date: 2023-09-08T17:01:53Z
#url: https://api.github.com/gists/699fdbc72d777222ea61fd39461a0563
#owner: https://api.github.com/users/kaznak

# MIT License
from transformers import AutoTokenizer
import transformers
from langchain.document_loaders import PyPDFLoader
import torch

model = "NousResearch/Yarn-Llama-2-13b-128k"

tokenizer = "**********"
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)


loader = PyPDFLoader("/path/to/paper")
documents = loader.load()

print(len(documents))
document=""
for doc in documents:
    document+=doc.page_content
text=document.replace("\n","")
print(len(text))

question="I am going to summarize the academic contribution of this paper in the following statement."
sequences = pipeline(
    f"I am going to read the following academic paper. \n\n {text} \n\n {question}\n",
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id= "**********"
    max_length=20000,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")"Result: {seq['generated_text']}")