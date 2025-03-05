#date: 2025-03-05T17:06:35Z
#url: https://api.github.com/gists/279ed5999f20dfe4bfd9efc67cd3131e
#owner: https://api.github.com/users/tejaswi-2230

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from models.website import Website
from schemas.website import WebCreate, WebResponse
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

@router.get("/sites/", response_model=list[WebResponse])
def get_website_URLs(db: Session = Depends(get_db)):
    return db.query(Website).all()

@router.post("/sites/", response_model=WebResponse)
def create_site(site: "**********": Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        new_site = db.query(Website).filter(Website.url==site.url).first()
        if new_site:
            raise HTTPException(status_code=400,detail="URL already exists")
        sites = Website(url=site.url, description=site.description, interval = site.interval)
        sites.createdBy = cred[1]
        db.add(sites)
        db.commit()
        db.refresh(sites)
        return sites
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")


@router.put("/sites/{site_id}", response_model=WebResponse)
def update_URL(site_id: "**********": WebCreate, db: Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        existing_url = db.query(Website).filter(Website.web_id == site_id).first()
        if not existing_url:
            raise HTTPException(status_code=404, detail="url not found")

        existing_url.url = site.url
        existing_url.description = site.description
        existing_url.interval = site.interval
        existing_url.createdBy = cred[1]
        db.commit()
        db.refresh(existing_url)
        return existing_url
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")


@router.delete("/sitess/{site_id}", response_model=dict)
def delete_URL(site_id: "**********": Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        url = db.query(Website).filter(Website.web_id == site_id).first()
        if not url:
            raise HTTPException(status_code=404, detail="URL not found")
        db.delete(url)
        db.commit()
        return {"message": "URL deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")
