#date: 2025-03-05T17:06:35Z
#url: https://api.github.com/gists/279ed5999f20dfe4bfd9efc67cd3131e
#owner: https://api.github.com/users/tejaswi-2230

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from base import SessionLocal
from models.userorganisation import UserOrganisation
from schemas.userorganization import UserOrgCreate, UserOrgResponse
from auth.auth import decode_token

router = APIRouter()

roles = ["Admin", "admin"]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/get_user_organisation/", response_model=list[UserOrgResponse])
def get_userorgs(db: Session = Depends(get_db)):
    return db.query(UserOrganisation).all()

@router.post("/create_user_organisation/", response_model=UserOrgResponse)
def create_userorg(userorg: "**********": Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        exist_userorg = db.query(UserOrganisation).filter(UserOrganisation.user_id == userorg.user_id, UserOrganisation.org_id == userorg.org_id).first()
        if exist_userorg:
            raise HTTPException(status_code=400, detail="user id and organization id already exists")
        
        new_user = UserOrganisation(user_id=userorg.user_id, org_id=userorg.org_id)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")


@router.put("/update_user_organisation/{id}", response_model=UserOrgResponse)
def update_userorg(id: "**********": UserOrgCreate, db: Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        existing_userorg = db.query(UserOrganisation).filter(UserOrganisation.userorg_id == id).first()
        if not existing_userorg:
            raise HTTPException(status_code=404, detail="User / Organization not found")
        
        exist_userorg = db.query(UserOrganisation).filter(UserOrganisation.user_id == userorg.user_id, UserOrganisation.org_id == userorg.org_id).first()
        if exist_userorg:
            raise HTTPException(status_code=400, detail="user id and organization id already exists")
        
        existing_userorg.user_id = userorg.user_id
        existing_userorg.org_id = userorg.org_id
        db.commit()
        db.refresh(existing_userorg)
        return existing_userorg
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")


@router.delete("/delete_user_organisation/{id}", response_model=dict)
def delete_userorg(id: "**********": Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        userorg = db.query(UserOrganisation).filter(UserOrganisation.id == id).first()
        if not userorg:
            raise HTTPException(status_code=404, detail="User / Organization not found")
        db.delete(userorg)
        db.commit()
        return {"message": "User and/or Organization deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")
