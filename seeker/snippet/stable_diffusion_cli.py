#date: 2022-12-28T16:26:37Z
#url: https://api.github.com/gists/a0ebff5dc2bf23dead003f6f516bfc51
#owner: https://api.github.com/users/luiscape

import io
import os
import time
from pathlib import Path

import modal
import typer

stub = modal.Stub("stable-diffusion-cli")
app = typer.Typer()


model_id = "runwayml/stable-diffusion-v1-5"
cache_path = "/vol/cache"


def download_models():
    import diffusers
    import torch

    hugging_face_token = "**********"

    # Download scheduler configuration. Experiment with different schedulers
    # to identify one that works best for your use-case.
    scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
        model_id, subfolder= "**********"=hugging_face_token, cache_dir=cache_path
    )
    scheduler.save_pretrained(cache_path, safe_serialization=True)

    # Downloads all other models.
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        model_id, use_auth_token= "**********"="fp16", torch_dtype=torch.float16, cache_dir=cache_path
    )
    pipe.save_pretrained(cache_path, safe_serialization=True)


image = (
    # Use PyTorch image from the NVIDIA NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
    # NVIDIA DGC images perform better in benchmarks [1]. This is related to an issue with dynamic or static
    # linking of NVIDIA deep learning libraries used by PyTorch. [2]
    # [1] https://gist.github.com/rwightman/bb59f9e245162cee0e38bd66bd8cd77f
    # [2] https://github.com/pytorch/pytorch/issues/50153#issuecomment-808854369
    modal.Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "accelerate",
        "diffusers[torch]>=0.10",
        "ftfy",
        "torch",
        "torchvision",
        "transformers",
        "triton",
        "safetensors",
        "xformers==0.0.16rc393",
        "tensorboard==2.11.0",
    )
    .run_function(
        download_models,
        secrets= "**********"
    )
)
stub.image = image


class StableDiffusion:
    def __enter__(self):
        start = time.time()
        import diffusers
        import torch
        print(f"imports => {time.time() - start:.3f}s")

        torch.backends.cuda.matmul.allow_tf32 = True

        time_load = time.time()
        scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
            cache_path,
            subfolder="scheduler",
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            denoise_final=True,  # important if steps are <= 10
        )
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(cache_path, scheduler=scheduler)
        print(f"load pipe => {time.time() - time_load:.3f}s")

        time_to_cuda = time.time()
        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()
        print(f"to cuda => {time.time() - time_to_cuda:.3f}s")

        print(f"init total => {time.time() - start:.3f}s")

    @stub.function(gpu="A10G")
    def run_inference(self, prompt: str, steps: int = 20, batch_size: int = 4):
        import torch

        start = time.time()
        with torch.inference_mode():
            with torch.autocast("cuda"):
                images = self.pipe([prompt] * batch_size, num_inference_steps=steps, guidance_scale=7.0).images

        print(f"inference => {time.time() - start:.3f}s")

        # Convert to PNG bytes
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        return image_output


@app.command()
def entrypoint(prompt: str, samples: int = 5, steps: int = 10, batch_size: int = 1):
    typer.echo(f"prompt => {prompt}, steps => {steps}, samples => {samples}, batch_size => {batch_size}")

    dir = Path("/tmp/stable-diffusion")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    with stub.run():
        sd = StableDiffusion()
        for i in range(samples):
            t0 = time.time()
            images = sd.run_inference.call(prompt, steps, batch_size)
            total_time = time.time() - t0
            print(f"Sample {i} took {total_time:.3f}s ({(total_time)/len(images):.3f}s / image).")
            for j, image_bytes in enumerate(images):
                output_path = dir / f"output_{j}_{i}.png"
                print(f"Saving it to {output_path}")
                with open(output_path, "wb") as f:
                    f.write(image_bytes)


if __name__ == "__main__":
    app()


                    f.write(image_bytes)


if __name__ == "__main__":
    app()

