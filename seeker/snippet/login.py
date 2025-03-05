#date: 2025-03-05T17:06:35Z
#url: https://api.github.com/gists/279ed5999f20dfe4bfd9efc67cd3131e
#owner: https://api.github.com/users/tejaswi-2230

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from models.users import User
from schemas.users import UserLogin
from base import SessionLocal
from auth.auth import create_token

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/login/", tags = ["Login"])
def user_login(user : UserLogin , db: Session = Depends(get_db)):
    user_new = db.query(User).filter(User.email == user.email).first()
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"u "**********"s "**********"e "**********"r "**********"_ "**********"n "**********"e "**********"w "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"u "**********"s "**********"e "**********"r "**********". "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"= "**********"= "**********"  "**********"u "**********"s "**********"e "**********"r "**********"_ "**********"n "**********"e "**********"w "**********". "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********": "**********"
        raise HTTPException(status_code= "**********"="Invalid email or password")
    
    access_token = "**********"
        "usermail": user.email,
        "role" : user_new.role,
        "id" : user_new.user_id
    })
    return {"access_token": "**********": "bearer"}