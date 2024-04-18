#date: 2024-04-18T16:46:29Z
#url: https://api.github.com/gists/9ed324ccfa5ffeb51c92b14ce15a2a64
#owner: https://api.github.com/users/agyaatcoder

#Meta-Llama-3-8B-Instruct is gated model and requires access on hf first to be able to successfully run this
import os
import subprocess
from modal import Image, Secret, Stub, gpu, web_server


MODEL_DIR = "/model"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DOCKER_IMAGE = "ghcr.io/huggingface/text-generation-inference:1.4"
PORT = 8000

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
]

def download_model():
    subprocess.run(
        [
            "text-generation-server",
            "download-weights",
            MODEL_ID,
        ],
        env={
            **os.environ,
            "HUGGING_FACE_HUB_TOKEN": "**********"
        },
        check=True,
    )
GPU_CONFIG = gpu.A100(memory=80)

stub = Stub("llama3-8b-instruct")

tgi_image = (
    Image.from_registry(DOCKER_IMAGE, add_python="3.10")
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model, timeout= "**********"=[Secret.from_name("hf-secret-llama")])
)


@stub.function(
    image=tgi_image,
    gpu=GPU_CONFIG,
    concurrency_limit= 10,
    secrets= "**********"
)
@web_server(port=PORT, startup_timeout=120)
def run_server():
    model = MODEL_ID
    port = PORT
    cmd = f"text-generation-launcher --model-id {model} --hostname 0.0.0.0 --port {port}"
    subprocess.Popen(cmd, shell=True)

#Once you receive your endpoint: https://xyz-modal-is-awesome.modal.run, you can consume in this way
# curl https://xyz-modal-is-awesome.modal.run/generate \
#     -X POST \
#     -d '{"inputs": "**********":{"max_new_tokens":20}}' \
#     -H 'Content-Type: application/json'   -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
#     -H 'Content-Type: application/json'