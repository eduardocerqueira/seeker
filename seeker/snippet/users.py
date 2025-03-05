#date: 2025-03-05T17:06:35Z
#url: https://api.github.com/gists/279ed5999f20dfe4bfd9efc67cd3131e
#owner: https://api.github.com/users/tejaswi-2230

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from models.users import User
from schemas.users import UserCreate, UserResponse
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

@router.get("/get_users/", response_model=list[UserResponse])
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@router.post("/add_users/", response_model=UserResponse)
def create_user(user: "**********": Session = Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        new_user = "**********"= user.first_name, last_name = user.last_name, email=user.email, password=user.password, role = user.role)
        new_user.createdBy = cred[1]
        db.add(new_user)
        try:
            db.commit()
            db.refresh(new_user)
            return new_user
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail="Email already exists")
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
