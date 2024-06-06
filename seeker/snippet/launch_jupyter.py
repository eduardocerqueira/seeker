#date: 2024-06-06T17:07:16Z
#url: https://api.github.com/gists/29495bf9f9703f9de77f92f480d9988e
#owner: https://api.github.com/users/aksh-at

import os
import secrets
import subprocess
import time

import modal

app = modal.App()

image = (
    modal.Image.debian_slim()
    .pip_install("jupyter", "bing-image-downloader~=1.1.2")
    .pip_install("pandas")  # Any other deps
)


@app.function(image=image, timeout=1_500)
def run_jupyter(queue, timeout=1_500):
    token = "**********"
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": "**********"
        )

        url = "**********"={token}"
        queue.put(url)

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main():
    with modal.Queue.ephemeral() as queue:
        run_jupyter.spawn(queue)
        tunnel_url = queue.get()
        print(f"Jupyter notebook is running at: {tunnel_url}")

    time.sleep(1000)
time.sleep(1000)
