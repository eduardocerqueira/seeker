#date: 2025-03-05T17:06:35Z
#url: https://api.github.com/gists/279ed5999f20dfe4bfd9efc67cd3131e
#owner: https://api.github.com/users/tejaswi-2230

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from models.project import Project
from schemas.project import ProjectCreate, ProjectRespose
from base import SessionLocal
from auth.auth import decode_token

router = APIRouter()

roles = ["Admin", "admin"]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/get_projects/", response_model=list[ProjectRespose])
def get_projects(db : Session = Depends(get_db)):
    return db.query(Project).all()

@router.post("/add_projects/", response_model=ProjectRespose)
def create_project(project : "**********": Session=Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        exist_project = db.query(Project).all()
        for pro in exist_project:
            if pro.org_id == project.org_id and pro.project_name == project.project_name:
                raise HTTPException(status_code= 400, detail="Project already exists")
        new_project = Project(org_id = project.org_id, project_name = project.project_name)
        new_project.createdBy = cred[1]
        db.add(new_project)
        try:
            db.commit()
            db.refresh(new_project)
            return new_project
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail=e)
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")
    
@router.put("/update_projects/{project_id}", response_model=ProjectRespose)
def update_project(project_id : "**********": ProjectCreate, db : Session=Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        exist_project = db.query(Project).filter(Project.project_id == project_id).first()
        if not exist_project:
            raise HTTPException(status_code=404, detail="Project does not exists")
        exist_project.org_id = project.org_id
        exist_project.project_name = project.project_name
        exist_project.createdBy = cred[1]
        db.commit()
        db.refresh(exist_project)
        return exist_project
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")
    
@router.delete("/delete_projects/{project_id}", response_model=dict)
def delete_project(project_id : "**********": Session=Depends(get_db), cred = Depends(decode_token)):
    if cred[0] in roles:
        exist_project = db.query(Project).filter(Project.project_id == project_id).first()
        if not exist_project:
            raise HTTPException(status_code=404, detail="Project does not exists")
        db.delete(exist_project)
        db.commit()
        return {"message" : "Project deleted Successfully"}
    else:
        raise HTTPException(status_code=404, detail="This user doest have access")s")