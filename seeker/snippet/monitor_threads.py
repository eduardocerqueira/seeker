#date: 2024-10-03T17:01:49Z
#url: https://api.github.com/gists/30c1c572e3c2403528e1a3ed612726ce
#owner: https://api.github.com/users/rajvermacas

import threading
import time
import gc
import logging as logger
import traceback
import sys
import os

# Set up logger
logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

# Add a global flag to control the monitor thread
monitor_flag = threading.Event()

def get_thread_info():
    thread_info = []
    for thread_id, frame in sys._current_frames().items():
        try:
            thread_obj = threading._active.get(thread_id)
            if thread_obj:
                thread_name = thread_obj.name
            else:
                thread_name = f"Thread-{thread_id}"

            # Get stack trace
            stack = traceback.extract_stack(frame)
            stack_info = "".join(traceback.format_list(stack))

            thread_info.append({
                'name': thread_name,
                'id': thread_id,
                'stack': stack_info
            })
        except Exception as e:
            thread_info.append({
                'name': f"Thread-{thread_id}",
                'id': thread_id,
                'error': str(e)
            })
    
    return thread_info

def monitor_threads():
    while monitor_flag.is_set():
        threads_info = get_thread_info()
        logger.info(f"Current running threads (Process ID: {os.getpid()}). Number of active threads={len(threads_info)}")
        thread_count = 0
        for info in threads_info:
            thread_count += 1

            logger.info(f"---------thread index={thread_count}-----------")
            logger.info(f"Thread: {info['name']} (ID: {info['id']})")

            if 'stack' in info:
                logger.info(f"Stack Trace:\n{info['stack']}")
            if 'error' in info:
                logger.info(f"Error: {info['error']}")

        logger.info("=======================================")
        time.sleep(5)  # Wait for 5 seconds before next check

def trigger_gc():
    time.sleep(10)  # Wait for 10 seconds before triggering GC
    gc.collect()
    logger.info("Garbage collection triggered")

def cpu_intensive_task():
    logger.info("Starting CPU intensive task")
    for _ in range(10**7):
        _ = [i**2 for i in range(100)]
    logger.info("CPU intensive task completed")

if __name__ == "__main__":
    start_time = time.time()
    # Start the monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_threads, name="MonitorThread")
    monitor_thread.daemon = True

    # Set the monitor flag to True to start monitoring
    monitor_flag.set()
    monitor_thread.start()

    # # Start a thread to trigger garbage collection
    # gc_thread = threading.Thread(target=trigger_gc, name="GCThread")
    # gc_thread.start()

    # # Start a CPU intensive task to demonstrate thread activity
    cpu_thread = threading.Thread(target=cpu_intensive_task, name="CPUIntensiveThread")
    cpu_thread.start()

    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
            # Example of how to stop the monitoring after 30 seconds
            if time.time() - start_time > 30:
                monitor_flag.clear()  # This will stop the monitor thread
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    finally:
        # Ensure the monitor thread is stopped
        monitor_flag.clear()
        monitor_thread.join()