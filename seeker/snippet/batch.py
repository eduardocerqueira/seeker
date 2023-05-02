#date: 2023-05-02T17:00:45Z
#url: https://api.github.com/gists/0ceed3e0c76c3ee8a8d16dbd66330742
#owner: https://api.github.com/users/growdexo

# Generic batch processing class
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from tqdm import tqdm

class Batch:
    def __init__(self) -> None:
          self.jobs = []
          
    def add_job(self, job):
        self.jobs.append(job)
        
    def run(self, num_workers):
        with tqdm(total=len(self.jobs)) as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit jobs to the executor, but they won't run immediately
                futures = [executor.submit(job) for job in self.jobs]
                for future in as_completed(futures):
                    result = future.result()
                    pbar.update(1)

                # Wait for all jobs to completed
                wait(futures)

    print("All jobs completed.")