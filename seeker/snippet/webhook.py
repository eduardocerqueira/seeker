#date: 2022-10-31T17:28:04Z
#url: https://api.github.com/gists/cf202a8d6d7d09f6dcaa26c44ee275a6
#owner: https://api.github.com/users/Shawnice

import hashlib
import hmac
import http
import os

from fastapi import FastAPI, Header, HTTPException, Request

app = FastAPI()


def generate_hash_signature(
    secret: "**********"
    payload: bytes,
    digest_method=hashlib.sha1,
):
    return hmac.new(secret, payload, digest_method).hexdigest()


@app.post("/webhook", status_code=http.HTTPStatus.ACCEPTED)
async def webhook(request: Request, x_hub_signature: str = Header(None)):
    payload = await request.body()
    secret = "**********"
    signature = "**********"
    if x_hub_signature != f"sha1={signature}":
        raise HTTPException(status_code=401, detail="Authentication error.")
    return {}us_code=401, detail="Authentication error.")
    return {}