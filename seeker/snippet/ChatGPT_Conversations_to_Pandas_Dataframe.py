#date: 2023-11-29T16:52:25Z
#url: https://api.github.com/gists/59813058cfe5d75cd90d891e3491134e
#owner: https://api.github.com/users/Cdaprod

import pandas as pd
import json

# Load the JSON data from 'conversations.json' in the current working directory
with open('conversations.json', 'r') as file:
    conversations = json.load(file)

# Extract and flatten the data
flattened_data = []
for conversation in conversations:
    for key, value in conversation['mapping'].items():
        if value.get('message') and value['message'].get('content'):
            text_parts = value['message']['content'].get('parts', [])
            text = ' '.join(text_parts)
            flattened_data.append({
                'id': value.get('id', key),
                'text': text,
                'source': value['message'].get('author', {}).get('role', 'unknown'),
                'metadata': value['message'].get('metadata', {})
            })

# Create DataFrame
df = pd.DataFrame(flattened_data)

# Export DataFrame to a format supported by Canopy (e.g., jsonl)
df.to_json('conversations.jsonl', orient='records', lines=True)
