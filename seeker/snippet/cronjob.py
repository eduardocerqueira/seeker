#date: 2025-03-05T17:06:35Z
#url: https://api.github.com/gists/279ed5999f20dfe4bfd9efc67cd3131e
#owner: https://api.github.com/users/tejaswi-2230

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from models.cronjob import CronJob
from schemas.cronjob import CronJobResponse, CronJobCreate
from models.website import Website
from base import SessionLocal
from auth.auth import decode_token
from apscheduler.triggers.interval import IntervalTrigger
from scripts.uptime_bot import check
from apscheduler.schedulers.background import BackgroundScheduler
import logging
from base import engine
from sqlalchemy import inspect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("website_status.log"), logging.StreamHandler() 
    ]
)

router = APIRouter()

roles = ['Admin', 'admin']

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

scheduler = BackgroundScheduler()
scheduler.start()

def add_job(link, timeout, job_id):
    trigger = IntervalTrigger(seconds=timeout)
    scheduler.add_job(check, trigger, args=[link, int(job_id[8:])], id=job_id)

def start_job():
    db = SessionLocal()
    active_jobs = db.query(CronJob).filter(CronJob.isExist == True).all()
    for job in active_jobs:
        web = db.query(Website).filter(Website.web_id == job.web_id).first()
        if web:
            job_id = f"cronjob_{job.cronjob_id}"  # Ensure unique job ID
            add_job(web.url, web.interval, job_id)
            logging.info(f"Job {job_id} restarted successfully")
    db.close()

inspector = inspect(engine)

if "cronjob" in inspector.get_table_names():
    start_job()

@router.get("/get_cronjob/", response_model=list[CronJobResponse])
def get_cronjob(db: Session = Depends(get_db)):
    return db.query(CronJob).all()

@router.post("/create_cronjob/", response_model=CronJobResponse)
async def create_cronjob(id: "**********": Session = Depends(get_db), cred=Depends(decode_token)):
    web = db.query(Website).filter(Website.web_id == id).first()
    if not web:
        raise HTTPException(status_code=404, detail="Website not found")

    job = db.query(CronJob).filter(CronJob.web_id == id).first()
    
    if job:
        if job.isExist == True:
            raise HTTPException(status_code=400, detail="Job is already running")
        else:
            job_id = f"cronjob_{job.cronjob_id}"
            add_job(web.url, web.interval, job_id)
            job.isExist = True
            db.commit()
            db.refresh(job)
            logging.info(f"Resuming job {job_id}")
    else:
        # Create a new job
        new_job = CronJob(web_id=id, isExist=True)
        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        job_id = f"cronjob_{new_job.cronjob_id}"
        add_job(web.url, web.interval, job_id)
        logging.info(f"Job {job_id} created successfully")

    return job if job else new_job

@router.delete("/delete_cronjob/{id}", response_model=CronJobResponse)
async def delete_cronjob(id: "**********": Session = Depends(get_db), cred=Depends(decode_token)):
    exist_job = db.query(CronJob).filter(CronJob.cronjob_id == id).first()
    if not exist_job or exist_job == False:
        raise HTTPException(status_code=404, detail="Job not found")

    # Mark job as inactive
    exist_job.isExist = False
    db.commit()
    db.refresh(exist_job)

    # Remove the scheduled job from APScheduler
    job_id = f"cronjob_{id}"
    if scheduler.get_job(job_id):  # Check if job exists before removing
        scheduler.remove_job(job_id)
        logging.info(f"Job {job_id} removed from scheduler")

    return exist_job
