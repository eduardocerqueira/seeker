#date: 2023-11-09T16:42:55Z
#url: https://api.github.com/gists/c9c068f66d371175e5334d1635deaea0
#owner: https://api.github.com/users/JacobFV

from fastapi import FastAPI, Depends
from celery import Celery
from sqlmodel import SQLModel, Field, create_engine, Session
from datetime import datetime, timedelta
from celery.result import AsyncResult
import time
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager

# Database setup for multi-tenant application
DATABASE_URL = "postgresql: "**********":password@localhost/mydatabase"
engine = create_engine(DATABASE_URL)

# Define the enum for task statuses
class TaskStatus(str, Enum):
    STARTED = "STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILURE = "FAILURE"
    REVOKED = "REVOKED"
    RESTARTED = "RESTARTED"

# Define Task model for database interactions
class TaskModel(SQLModel, table=True):
    __tablename__ = "tasks"
    id: int = Field(default=None, primary_key=True)
    task_id: str = Field(sa_column_kwargs={"unique": True, "index": True})
    tenant_id: str
    start_time: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = Field(sa_column_kwargs={"enum": SQLEnum(TaskStatus)})

# Create tables in the database
SQLModel.metadata.create_all(engine)

# Context manager for database sessions to ensure safe transactions
@contextmanager
def get_session():
    session = Session(engine)
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()

# Celery configuration using Redis for task queuing and result storage
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# BaseTask class to handle common database logic and heartbeat updates
class BaseTask(celery_app.Task):

    def before_start(self, tenant_id):
        # Create a new task record in the database at the start
        with get_session() as db:
            new_task = TaskModel(task_id=self.request.id, tenant_id=tenant_id, status=TaskStatus.STARTED)
            db.add(new_task)

    def update_heartbeat(self, tenant_id, progress_info):
        # Update task heartbeat (progress) in a consistent manner for all tasks
        self.update_state(
            state='PROGRESS',
            meta={'tenant_id': tenant_id, **progress_info}
        )

    def after_completion(self):
        # Update task status to 'COMPLETED' in the database
        with get_session() as db:
            task = db.get(TaskModel, self.request.id)
            if task:
                task.status = TaskStatus.COMPLETED

# Wait2HoursTask inherits from BaseTask and implements specific logic
class Wait2HoursTask(BaseTask):
    name = 'wait_2_hours_task'

    def run(self, tenant_id: str):
        try:
            self.before_start(tenant_id)
            
            # Simulate a long-running process (120 minutes)
            for minute in range(120):
                time.sleep(60)
                self.update_heartbeat(tenant_id, {'current_minute': minute, 'total_minutes': 120})

            self.after_completion()

        except Exception as e:
            logger.error(f"Error in Wait2HoursTask: {e}")
            raise

        return 'Completed 2 hours wait'

# Register the task with Celery
wait2hours_task = Wait2HoursTask()
celery_app.tasks.register(wait2hours_task)

# Task for monitoring and restarting tasks
@celery_app.task
def monitor_tasks():
    with get_session() as db:
        # Query for ongoing tasks from the database
        tasks = db.query(TaskModel).filter(TaskModel.status == TaskStatus.STARTED).all()

        for task in tasks:
            # Check the current state of the task
            task_result = AsyncResult(task.task_id, app=celery_app)

            # Handle failed or revoked tasks
            if task_result.state in ['FAILURE', 'REVOKED']:
                task.status = TaskStatus.FAILURE

            # Check if ongoing tasks need intervention (e.g., restarting)
            elif task_result.state == 'STARTED':
                if needs_intervention(task, task_result):
                    restart_task(task.task_id)
                    task.status = TaskStatus.RESTARTED

# Determine if a task needs intervention based on duration
def needs_intervention(task, task_result):
    max_duration = timedelta(minutes=150)
    if datetime.now() - task.start_time > max_duration:
        return True
    return False

# Function to restart a task
def restart_task(task_id):
    logger.info(f"Restarting task {task_id}")
    celery_app.control.revoke(task_id, terminate=True)

    new_task = wait2hours_task.apply_async()
    with get_session() as db:
        task = db.query(TaskModel).filter(TaskModel.task_id == task_id).first()
        if task:
            task.task_id = new_task.id

# FastAPI app initialization
app = FastAPI()

# Endpoint to start the long-running task
@app.post("/wait2hours/{tenant_id}")
def start_wait2hours_task(tenant_id: str, session: Session = Depends(get_session)):
    task = wait2hours_task.apply_async(args=[tenant_id])
    return {"task_id": task.id, "tenant_id": tenant_id}

# Main block to run the FastAPI server for a multi-tenant, fault-tolerant application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
