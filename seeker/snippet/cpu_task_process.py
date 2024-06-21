#date: 2024-06-21T16:58:59Z
#url: https://api.github.com/gists/0baba7ba54dee22ee4085113b50c41fa
#owner: https://api.github.com/users/vndee

from multiprocessing import Process

def cpu_bound_task():
    count = 0
    for _ in range(10**7):
        count += 1
    print("CPU-bound task complete")

processes = []
for _ in range(5):
    process = Process(target=cpu_bound_task)
    processes.append(process)
    process.start()

for process in processes:
    process.join()
