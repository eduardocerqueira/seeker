#date: 2023-04-14T16:48:26Z
#url: https://api.github.com/gists/f3703c0cc2d58010faf8a67bd5552b29
#owner: https://api.github.com/users/jason19970210

from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

## Init
app = FastAPI()
scheduler = BackgroundScheduler()

def timer() -> None:
    print(datetime.now().strftime('%Y%m%d-%H:%M:%S'))

def ret_jobs_id(scheduler: BackgroundScheduler) -> list:
    return  [ i.id for i in scheduler.get_jobs() ]

## backend
@app.get('/jobs')
def get_jobs():
    ret_dict = {}
    ret_dict['jobs'] = ret_jobs_id(scheduler=scheduler)
    return JSONResponse(content=ret_dict)

@app.post('/jobs/{job_id}')
def add_jobs(job_id: str):
    ret_dict = {}
    scheduler.add_job(func=timer, trigger='interval', seconds=1, id=job_id)
    ret_dict['jobs'] = ret_jobs_id(scheduler=scheduler)
    return JSONResponse(content=ret_dict)

@app.delete('/jobs/{job_id}')
def del_jobs(job_id: str):
    ret_dict = {}
    scheduler.remove_job(job_id=job_id)
    ret_dict['jobs'] = ret_jobs_id(scheduler=scheduler)
    return JSONResponse(content=ret_dict)

# main
if __name__ == '__main__':
    scheduler.start()
    uvicorn.run(app, host='0.0.0.0', port=8000)