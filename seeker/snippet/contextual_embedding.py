#date: 2025-03-05T16:41:51Z
#url: https://api.github.com/gists/15ff95d3b66510232ba2d44daab45fe5
#owner: https://api.github.com/users/vhoudebine

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

aoai_client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_KEY"),  
  api_version = "2024-02-01",
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def llm_call(client, model, context, chunk):
    prompt = f"""<document> 
{context} 
</document> 
Here is the chunk we want to situate within the document 
<chunk> 
{chunk} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
If the chunk contains table cells, please provide a summary of the missing table cell values and headers.
"""
    messages = [
        {"role": "system", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens= "**********"
    )

    return response.choices[0].message.content
    

neighboring_documents = 20

enriched_chunks = []

for i, chunk in enumerate(filtered_chunks):
    start = max(0, i - neighboring_documents // 2)
    end = min(len(filtered_chunks), i + neighboring_documents // 2 + 1)
    neighboring_chunks = filtered_chunks[start:end]
    aggregated_content = " ".join([neighbor.page_content for neighbor in neighboring_chunks])
    
    contextual_summary = llm_call(aoai_client, deployment, aggregated_content, chunk.page_content)
    contextual_chunk = f"{contextual_summary}\n\n{chunk.page_content}"

    enriched_chunks.append({
        "chunk": chunk.page_content,
        "contextual_chunk": contextual_chunk,
        "chunk_number": i,
        "metadata": chunk.metadata
    })