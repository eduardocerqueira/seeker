#date: 2024-02-09T16:57:19Z
#url: https://api.github.com/gists/6fe398ab0c223d8bfa373a9383417780
#owner: https://api.github.com/users/eiAlex

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI()

# Secret key to sign JWT tokens
SECRET_KEY = "**********"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = "**********"

# Fake user data for demonstration purposes
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "hashed_password": "**********"
    }
}

# Blacklist to store revoked tokens
token_blacklist = "**********"

 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********"D "**********"a "**********"t "**********"a "**********": "**********"
    def __init__(self, username: str | None = None):
        self.username = username

oauth2_scheme = "**********"="token")

# Function to create JWT tokens
def create_access_token(data: "**********": timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = "**********"=ALGORITHM)
    return encoded_jwt

# Function to get current user from token
def get_current_user(token: "**********":
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = "**********"=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = "**********"=username)
    except JWTError:
        raise credentials_exception

    # Check if the token is in the blacklist
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"b "**********"l "**********"a "**********"c "**********"k "**********"l "**********"i "**********"s "**********"t "**********": "**********"
        raise credentials_exception

    return token_data

# Endpoint to get a token
@app.post("/token")
async def login_for_access_token(form_data: "**********":
    user = fake_users_db.get(form_data.username)
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"u "**********"s "**********"e "**********"r "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********"  "**********"o "**********"r "**********"  "**********"f "**********"o "**********"r "**********"m "**********"_ "**********"d "**********"a "**********"t "**********"a "**********". "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"! "**********"= "**********"  "**********"" "**********"t "**********"e "**********"s "**********"t "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"" "**********": "**********"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail= "**********"
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = "**********"=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": "**********"
    return {"access_token": "**********": "bearer"}

# Endpoint to logout (revoke) a token
@app.post("/logout")
async def logout(current_user: "**********":
    # Add the token to the blacklist
    token_blacklist.add(current_user.username)
    return {"message": "Logout successful"}

# Protected endpoint that requires authentication
@app.get("/users/me", response_model=dict)
async def read_users_me(current_user: "**********":
    return {"username": current_user.username}
