#date: 2024-04-03T16:54:34Z
#url: https://api.github.com/gists/a3fa31d42f61bd35008cf75b62ab0f87
#owner: https://api.github.com/users/shresthapradip

from fastapi import FastAPI
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def func():
    logger.info(f"request / endpoint!")
    return {"message": "hello world!"}