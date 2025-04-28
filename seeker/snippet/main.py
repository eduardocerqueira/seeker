#date: 2025-04-28T16:47:11Z
#url: https://api.github.com/gists/a70c54ba0b017ecc771840306a2cc4f4
#owner: https://api.github.com/users/AlirezaSaffariyan

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()
tasks_db = []


class Task(BaseModel):
    title: str
    description: str | None = None
    done: bool = False


@app.get("/")
def home():
    return {"message": "Welcome to the To-Do List API!"}


@app.post("/tasks")
def create_task(task: Task):
    tasks_db.append(vars(task))
    return {"message": "Task created!", "task": task}


@app.get("/tasks")
def get_all_tasks():
    return {"tasks": tasks_db}


@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    if task_id < 0 or task_id >= len(tasks_db):
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks_db[task_id]


@app.put("/tasks/{task_id}")
def update_task(task_id: int, updated_task: Task):
    if task_id < 0 or task_id >= len(tasks_db):
        raise HTTPException(status_code=404, detail="Task not found")

    tasks_db[task_id] = vars(updated_task)
    return {"message": "Task updated!", "task": tasks_db[task_id]}


@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):
    if task_id < 0 or task_id >= len(tasks_db):
        raise HTTPException(status_code=404, detail="Task not found")

    deleted_task = tasks_db.pop(task_id)
    return {"message": "Task deleted!", "task": deleted_task}
