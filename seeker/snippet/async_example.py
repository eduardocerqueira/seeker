#date: 2025-10-01T17:03:06Z
#url: https://api.github.com/gists/295ff6122c75cb54b7f50a271ccb734b
#owner: https://api.github.com/users/viz-prakash

import asyncio
import random
import time

# 1. Define an async function
async def fetch_data(task_id):
    """Simulates a task that takes a random amount of time."""
    delay = random.uniform(0.5, 2.0)
    print(f"Task {task_id}: Starting fetch with {delay:.2f}s delay.")
    await asyncio.sleep(delay)
    print(f"Task {task_id}: Finished fetch.")
    return f"Result from task {task_id} after {delay:.2f}s"

async def wrapper(num_tasks):
    """Creates and runs multiple async tasks, returning the results."""
    # Create a dynamic list of coroutine objects
    coroutines = (fetch_data(i) for i in range(1, num_tasks + 1))
    result = await asyncio.gather(*coroutines)
    return result

# 2. Define a non-async "main" function
def run_all_async_tasks(num_tasks):

    # Use asyncio.run() to execute the coroutines and wait for completion
    results = asyncio.run(wrapper(num_tasks))
    return results

# Example usage in a synchronous context
if __name__ == "__main__":
    num_tasks = 5
    print(f"Running {num_tasks} async tasks concurrently...")
    start_time = time.time()

    # Call the synchronous function to trigger the async workflow
    all_results = run_all_async_tasks(num_tasks)
    for res in all_results:
        print(res)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n--- All Tasks Completed ---")
    print(f"Final results: {all_results}")
    print(f"Time taken: {elapsed_time:.2f} seconds")