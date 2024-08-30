#date: 2024-08-30T17:12:04Z
#url: https://api.github.com/gists/22d5da6fb48207927a0dc4b8ec62df0d
#owner: https://api.github.com/users/zsasko

import json
from typing import AsyncGenerator, NoReturn

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI

load_dotenv()

model = "gpt-3.5-turbo"
conversation_history = []

app = FastAPI()
client = AsyncOpenAI()

with open("index.html") as f:
    html = f.read()
