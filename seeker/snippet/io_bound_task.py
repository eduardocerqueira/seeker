#date: 2024-06-21T16:53:52Z
#url: https://api.github.com/gists/57aaa38f78be2e22c8f6734130eedfbb
#owner: https://api.github.com/users/vndee

import time

# Simulate an I/O-bound task
def read_file():
    time.sleep(2)  # Simulating waiting for I/O operation (e.g., reading a file)
    return "File content"

# Calling the I/O-bound task
start_time = time.time()
content = read_file()
end_time = time.time()

print(f"Read file in {end_time - start_time} seconds")
