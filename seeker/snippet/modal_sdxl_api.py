#date: 2023-10-30T16:55:48Z
#url: https://api.github.com/gists/ed0a2bee8d1826704fe9f602490c3125
#owner: https://api.github.com/users/chand1012

from pathlib import Path

from modal import Image, Mount, Stub, asgi_app, gpu, method
from pydantic import BaseModel


def download_models():
    from huggingface_hub import snapshot_download

    ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
    snapshot_download(
        "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
    )
    snapshot_download(
        "stabilityai/stable-diffusion-xl-refiner-1.0", ignore_patterns=ignore
    )


image = (
    Image.debian_slim()
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers~=0.19",
        "invisible_watermark~=0.1",
        "transformers~=4.31",
        "accelerate~=0.21",
        "safetensors~=0.3",
    )
    .run_function(download_models)
)

stub = Stub("stable-diffusion-xl", image=image)


@stub.cls(gpu=gpu.T4(), container_idle_timeout=240)
class Model:
    def __enter__(self):
        import torch
        from diffusers import DiffusionPipeline

        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

        # Load refiner model
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

        # Compiling the model graph is JIT so this will increase inference time for the first run
        # but speed up subsequent runs. Uncomment to enable.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    @method()
    def inference(self, prompt, negative_prompt="disfigured, ugly, deformed", n_steps=24, high_noise_frac=0.8):
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        import io

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG", quality=85)
        image_bytes = byte_stream.getvalue()

        return image_bytes


@stub.local_entrypoint()
def main(prompt: str):
    image_bytes = Model().inference.remote(prompt)

    dir = Path(".")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)


class DiffusionReq(BaseModel):
    prompt: str
    negative_prompt: str = "disfigured, ugly, deformed"
    n_steps: int = 24


@stub.function()
@asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    import base64
    app = FastAPI()

    @app.post("/generate_image/")
    def generate_image(req: DiffusionReq):
        image_bytes = Model().inference.remote(
            req.prompt, req.negative_prompt, req.n_steps
        )
        # encode image as base64
        image_bytes = base64.b64encode(image_bytes)
        return {'b64_json': image_bytes}

    return app
