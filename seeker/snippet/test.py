#date: 2024-07-15T16:44:40Z
#url: https://api.github.com/gists/09b66e5290d0765cdb0019073fc5db08
#owner: https://api.github.com/users/LajnaLegenden

import requests
import time
import matplotlib.pyplot as plt
from collections import deque
import signal
import concurrent.futures
import threading

url = ""

results = deque(maxlen=100)  # Store up to 100 data points
interval = 10  # Time interval in seconds
pause_time = 5  # Pause time between intervals
max_concurrent_requests = 10  # Maximum number of concurrent requests

request_count = 0
request_count_lock = threading.Lock()
stop_event = threading.Event()

def signal_handler(sig, frame):
    print("Interrupt received. Saving graph...")
    save_graph()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

def save_graph():
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(results) + 1), results)
    plt.title("Requests per 10-second Interval")
    plt.xlabel("Interval Number")
    plt.ylabel("Number of Requests")
    plt.grid(True)
    plt.savefig("request_performance.png")
    print("Graph saved as request_performance.png")

def make_request(url):
    global request_count
    if not stop_event.is_set():
        try:
            print(f"Requesting URL: {url}")
            requests.get(url, timeout=9.5)  # Set timeout to slightly less than interval
            with request_count_lock:
                request_count += 1
        except requests.exceptions.RequestException:
            print("Request timed out or failed")

try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        while True:
            stop_event.clear()
            request_count = 0
            start_time = time.time()

            futures = []
            while time.time() - start_time < interval:
                futures.append(executor.submit(make_request, url))
                time.sleep(0.1)  # Small sleep to prevent CPU overuse

            stop_event.set()  # Signal threads to stop

            # Cancel any pending futures
            for future in futures:
                future.cancel()

            # Wait for all futures to complete or be cancelled
            concurrent.futures.wait(futures, timeout=0.5)

            results.append(request_count)
            print(f"Completed {request_count} requests in {interval} seconds")
            
            print(f"Pausing for {pause_time} seconds...")
            time.sleep(pause_time)

except KeyboardInterrupt:
    print("Interrupt received. Saving graph...")
    save_graph()