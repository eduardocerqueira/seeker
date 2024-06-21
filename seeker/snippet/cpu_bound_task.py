#date: 2024-06-21T16:53:23Z
#url: https://api.github.com/gists/b8f77fca12b969c7bbf8760f1581c982
#owner: https://api.github.com/users/vndee

import time

# Simulate a CPU-bound task
def compute_factorial(n):
    if n == 0:
        return 1
    else:
        return n * compute_factorial(n-1)

# Calling the CPU-bound task
start_time = time.time()
result = compute_factorial(20)
end_time = time.time()

print(f"Computed factorial in {end_time - start_time} seconds")
