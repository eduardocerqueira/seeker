#date: 2024-08-22T17:02:51Z
#url: https://api.github.com/gists/25346ca99e95a509aa8b5e374818253a
#owner: https://api.github.com/users/atiabjobayer

import requests
import json

# API Endpoint and Token
ACCESS_TOKEN = "**********"
CLIENT_ID = "saas_stern_trisso_com"
API_ENDPOINT = "https://askrobot.azurewebsites.net"

response = requests.post(
    API_ENDPOINT,
    headers={
        "Content-Type": "application/json",
        "Authorization": "**********"
    },
    json={
        "api": True,
        "engine": "answer",  # Use "answer" for RAG, "search" for searching
        "client": CLIENT_ID,
        "question": "What does God want for me?",  # Your natural language query
    },
)

# Print the raw response and processed data
print(response.text)
response_json_full = json.loads(response.text)oads(response.text)