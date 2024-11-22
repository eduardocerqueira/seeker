#date: 2024-11-22T16:49:30Z
#url: https://api.github.com/gists/cd084808affcf68bd3d9bcc410bebb51
#owner: https://api.github.com/users/apocas

import asyncio
import websockets
import json
import requests
import os
from collections import Counter
import re
from dotenv import load_dotenv

load_dotenv()

def restai(message):
    response = requests.post('https://ai.ince.pt/projects/bluesky/question', json={'question': message}, headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {os.environ.get("RESTAI_KEY")}'}, timeout=300)

    if response.status_code == 200:
        response_data = response.json()
        return response_data["answer"]

def display(categories, posts):
    top_words = categories.most_common(10)
    max_lines = max(len(top_words), len(posts))
    print("=" * 100)
    print(f"{'Top Categories':<28} {'|':^3} {'Realtime Posts':<68}")
    print("=" * 100)
    for i in range(max_lines):
        left = f"{i + 1}. {top_words[i][0]:<15} {top_words[i][1]:<5}" if i < len(top_words) else " " * 20
        right = posts[i] if i < len(posts) else ""
        print(f"{left:<28} {'|':^3} {right:<68}")
    print("=" * 100)

async def bluesky():
    posts = []
    categories = Counter()
    url = "wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"

    async with websockets.connect(url) as websocket:
        print("Connected to Bluesky Firehose!")

        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if "commit" in data and "record" in data["commit"] and "text" in data["commit"]["record"] and len(data["commit"]["record"]["text"].strip()) > 10:
                text = data["commit"]["record"]["text"].strip()
                category = restai(text).lower()
                
                if category and re.match(r'^\w+$', category):
                    categories[category] += 1
                    
                    if len(posts) > 10:
                        posts.pop(0)
                        
                    text = text.replace('\n', '')
                    if len(text) > 50:
                        text = text[:50] + "..."
                    posts.append("(" + category + ") " + text)

            os.system("cls" if os.name == "nt" else "clear")
            display(categories, posts)


asyncio.run(bluesky())
