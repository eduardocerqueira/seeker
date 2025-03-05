#date: 2025-03-05T17:06:35Z
#url: https://api.github.com/gists/279ed5999f20dfe4bfd9efc67cd3131e
#owner: https://api.github.com/users/tejaswi-2230

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from models.statsrequests import StatsRequests
from models.website import Website
from schemas.statsrequests import StatsRequestsResponse, StatsRequestsCreate
from base import SessionLocal
from auth.auth import decode_token
import os
import json

router = APIRouter()

roles = ['Admin', 'admin']

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/get_users/", response_model=list[StatsRequestsResponse])
def get_users(db: Session = Depends(get_db)):
    return db.query(StatsRequests).all()

@router.post("/add_users/", response_model=StatsRequestsResponse)
def create_user(web: "**********": Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        web_exist = db.query(Website).filter(Website.id == web.web_id).first()
        if not web_exist:
            raise HTTPException(status_code=404, detail="Website not found")
        output = os.popen(f"locust -f locustfile.py -u {web.no_of_users} -r 1 -t 10s --host={web_exist.url} --web-host=127.0.0.1 --autostart --skip-log --json --autoquit 2").read()
        json_output = json.loads(output)
        if json_output:
            
        return {}
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")

@router.put("/update_users/{user_id}", response_model=UserResponse)
def update_user(user_id: "**********": UserCreate, db: Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        existing_user = db.query(User).filter(User.user_id == user_id).first()
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")

        existing_user.first_name = user.first_name
        existing_user.last_name = user.last_name
        existing_user.email = user.email
        existing_user.password = "**********"
        existing_user.role = user.role
        existing_user.createdBy = cred[1]
        db.commit()
        db.refresh(existing_user)
        return existing_user
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")

@router.delete("/delete_users/{user_id}", response_model=dict)
def delete_user(user_id: "**********": Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        db.delete(user)
        db.commit()
        return {"message": "User deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")
s")
