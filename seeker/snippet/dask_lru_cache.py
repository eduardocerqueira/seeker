#date: 2024-08-19T16:59:37Z
#url: https://api.github.com/gists/29643a65bfdbf07cb5fa5e082f362087
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

    @lru_cache
    def make_task(task_id):
        deps = tasks[task_id]["dependsOn"]
        return execute_task(task_id, [make_task(dep_id) for dep_id in deps])

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
