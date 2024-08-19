#date: 2024-08-19T17:01:18Z
#url: https://api.github.com/gists/9754149ceee229670e8b3bba144c573b
#owner: https://api.github.com/users/andrii-i

from functools import lru_cache
import dask
from distributed import Client, LocalCluster


@dask.delayed(pure=True)
def execute_task(task_id: str, dependencies: list[str]):
    if dependencies:
        print(f"Task {task_id} executed with dependencies on {dependencies}")
    else:
        print(f"Task {task_id} executed without dependencies")
    return task_id


def execute(tasks):
    tasks = {task["id"]: task for task in tasks}

    cache = {}

    def make_task(task_id):
        try:
            return cache[task_id]
        except KeyError:
            deps = tasks[task_id]["dependsOn"]
            task = execute_task(task_id, [make_task(dep_id) for dep_id in deps])
            cache[task_id] = task
            return task

    final_tasks = [make_task(task_id) for task_id in tasks]
    print("Final tasks:")
    print(final_tasks)
    print(f"Calling compute after loop")
    return dask.compute(*final_tasks)


# Hardcoded tasks data
tasks_data = [
    {"id": "task0", "dependsOn": ["task3"]},
    {"id": "task1", "dependsOn": []},
    {"id": "task2", "dependsOn": ["task1"]},
    {"id": "task3", "dependsOn": ["task1", "task2"]},
]

if __name__ == "__main__":
    with LocalCluster(processes=True) as cluster:
        with Client(cluster) as client:
            results = execute(tasks_data)
            for result in results:
                print(result)
