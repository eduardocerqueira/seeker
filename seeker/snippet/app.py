#date: 2024-02-21T17:02:31Z
#url: https://api.github.com/gists/bc1ade1bef4fe7ec5027f7e9a0f28535
#owner: https://api.github.com/users/mvandermeulen

import typing as t

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from starlette import status


app = FastAPI()
# Placeholder for a database containing valid token values
known_tokens = "**********"
# We will handle a missing token ourselves
get_bearer_token = "**********"=False)

class UnauthorizedMessage(BaseModel):
    detail: "**********"


async def get_token(
    auth: "**********"
) -> str:
    # Simulate a database query to find a known token
    if auth is None or (token : "**********":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=UnauthorizedMessage().detail,
        )
    return token


@app.get(
    "/protected",
    response_model=str,
    responses={status.HTTP_401_UNAUTHORIZED: dict(model=UnauthorizedMessage)},
)
async def protected(token: "**********":
    return f"Hello, user! Your token is {token}."IZED: "**********"
)
async def protected(token: "**********":
    return f"Hello, user! Your token is {token}."llo, user! Your token is {token}."