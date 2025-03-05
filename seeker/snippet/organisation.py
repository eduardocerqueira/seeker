#date: 2025-03-05T17:06:35Z
#url: https://api.github.com/gists/279ed5999f20dfe4bfd9efc67cd3131e
#owner: https://api.github.com/users/tejaswi-2230

from fastapi import HTTPException, Depends, APIRouter
from models.organisation import Organisation
from schemas.organisation import OrgCreate, OrgResponse
from sqlalchemy.orm import Session
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

@router.get("/get_organisations/", response_model=list[OrgResponse])
def get_org(db : Session = Depends(get_db)):
    return db.query(Organisation).all()

@router.post("/add_organisations/", response_model=OrgResponse)
def create_org(org : "**********": Session = Depends(get_db), cred = Depends(decode_token)): 
    if cred[0] in roles:
        check_org = db.query(Organisation).all()
        check = False
        for orga in check_org:
            if orga.org_name == org.org_name and orga.org_address == org.org_address:
                check = True
                break
        if check:
            raise HTTPException(status_code= 400, detail="Organisation already exists")
        new_org = Organisation(org_name = org.org_name, org_address = org.org_address)
        new_org.createdBy = cred[1]
        db.add(new_org)
        try:
            db.commit()
            db.refresh(new_org)
            return new_org
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail=e)
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")
    
@router.put("/update_organisation/{org_id}", response_model=OrgResponse)
def update_org(org_id : "**********": OrgCreate, db : Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        exist_org = db.query(Organisation).filter(Organisation.org_id == org_id).first()
        if not exist_org:
            raise HTTPException(status_code=400, detail="Organisation not found")
        check_org = db.query(Organisation).all()
        check = False
        for orga in check_org:
            if orga.org_name == new_org.org_name and orga.org_address == new_org.org_address:
                check = True
        if check:
            raise HTTPException(status_code= 400, detail="Organisation already exists")
        exist_org.org_name = new_org.org_name
        exist_org.org_address = new_org.org_address
        exist_org.createdBy = cred[1]
        db.commit()
        db.refresh(exist_org)
        return exist_org
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")

@router.delete("/delete_organisation/{org_id}")
def delete_org(org_id : "**********": Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        exist_org = db.query(Organisation).filter(Organisation.org_id == org_id).first()
        if not exist_org:
            raise HTTPException(status_code=400, detail="Organisation not found")
        db.delete(exist_org)
        db.commit()
        return {"Message" : "Organisation deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")
