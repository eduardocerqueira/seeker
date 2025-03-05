#date: 2025-03-05T17:06:35Z
#url: https://api.github.com/gists/279ed5999f20dfe4bfd9efc67cd3131e
#owner: https://api.github.com/users/tejaswi-2230

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from models.webstats import WebStats
from schemas.webstats import WebStatsResponse
from base import SessionLocal
from auth.auth import decode_token


router = APIRouter()

roles = ['Admin', 'admin']

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/get_stats/", response_model=list[WebStatsResponse])
def get_website_URLs(db: "**********":
    return db.query(WebStats).all()
oken)):
    return db.query(WebStats).all()
