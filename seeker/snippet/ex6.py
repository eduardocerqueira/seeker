#date: 2024-07-01T16:58:13Z
#url: https://api.github.com/gists/0d6a9de044a81a78d04cc630b16f8fda
#owner: https://api.github.com/users/JonathanLoscalzo

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)

response = client.chat.completions.create(
    model="gemma:2b",
    messages=[{"role": "user", "content": "What's the formula for energy?"}],
    temperature=0.0,
)

print(response.usage) # CompletionUsage(completion_tokens= "**********"=34, total_tokens=317)=317)