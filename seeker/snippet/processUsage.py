#date: 2023-12-13T16:50:12Z
#url: https://api.github.com/gists/3609c5a67ede35128261a52af58625dd
#owner: https://api.github.com/users/Dsobh

import psutil
import GPUtil
import csv
import time
import sys


def monitor_process_usage(pid, interval=1):
    # csv file creation with write mode
    with open('process_monitoring.csv', 'w', newline='') as csvfile:
        # Definition of fields in the csv
        columnNames = ['Timestamp',
                       'CPU Usage (%)', 'Memory Usage (MB)', 'GPU Usage (%)']
        # Writer that operates mapping dictionaries into oputput rows.
        writer = csv.DictWriter(csvfile, fieldnames=columnNames)
        writer.writeheader()

        process = psutil.Process(pid)
        try:

            # Loop of execution until ctrl+c
            while True:
                # Get timestamp
                timestamp = time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.localtime())

                cpu_usage = process.cpu_percent(interval=interval)

                # Get memory usage (resident set size is in bytes) in mb
                memory_info = process.memory_info()
                memory_usage = memory_info.rss / (1024 * 1024)

                gpu_usage = 0.0
                try:
                    gpu_info = GPUtil.getGPUs()[0]
                    # GPU load (is from 0 to 1) in percentage
                    gpu_usage = gpu_info.load * 100
                except Exception as e:
                    print(f"Error getting GPU usage: {e}")

                # Write new row, using the fieldnames specified in writer creation
                writer.writerow({
                    'Timestamp': timestamp,
                    'CPU Usage (%)': cpu_usage,
                    'Memory Usage (MB)': memory_usage,
                    'GPU Usage (%)': gpu_usage
                })

                time.sleep(interval)
        except KeyboardInterrupt:
            print("Monitoring stopped.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        process_pid = int(sys.argv[1])
        interval = int(sys.argv[2])

    monitor_process_usage(process_pid, interval)
