#date: 2024-03-20T16:56:26Z
#url: https://api.github.com/gists/1deb90b2a5a0ce018b80bcfdf8ee277d
#owner: https://api.github.com/users/darrenangle

import subprocess
import time
import re
import signal
import sys
import select
import os

def start_server():
    command = [
        "/usr/bin/python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "hf-models/NousResearch-Hermes-2-Pro-Mistral-7B",
        "--max-model-len", "8192",
        "--enforce-eager"
    ]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid)

def read_output(process):
    error_pattern = re.compile(r"AsyncEngineDeadError: Task finished unexpectedly\.")
    while True:
        ready, _, _ = select.select([process.stdout, sys.stdin], [], [])
        if process.stdout in ready:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                if error_pattern.search(output):
                    print("Error detected. Restarting the server...")
                    terminate_process(process)
                    return True
        if sys.stdin in ready:
            input()  # Consume the input to prevent blocking
            print("Keyboard interrupt received. Terminating the server...")
            terminate_process(process)
            return False
    return False

def terminate_process(process):
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    time.sleep(5)  # Wait for a short duration to allow the process to terminate
    if process.poll() is None:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)

def main():
    while True:
        process = start_server()
        restart = read_output(process)
        if not restart:
            break
        time.sleep(5)  # Wait for 5 seconds before restarting

if __name__ == "__main__":
    main()