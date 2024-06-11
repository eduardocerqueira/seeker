#date: 2024-06-11T17:11:58Z
#url: https://api.github.com/gists/d94da75e9b8d47c40e9d29872af00e00
#owner: https://api.github.com/users/elijahbenizzy

import runhouse as rh
from diffusers import StableDiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-2-base"

# My function
def sd_generate(prompt, **inference_kwargs):
    model = StableDiffusionPipeline.from_pretrained(MODEL_ID)
    model = model.to("cuda")
    return model(prompt, **inference_kwargs).images


if __name__ == "__main__":
    # The compute
    gpu = rh.cluster(name="rh-a10x-a",
                     instance_type="A10G:1",
                     provider="aws").up_if_not()

    # The environment, its own process on the cluster
    sd_env = rh.env(reqs=["torch", "transformers", "diffusers", "accelerate"])

    # Deploying my function to my env on the cluster
    remote_sd_generate = rh.function(sd_generate).to(gpu, env=sd_env)

    # Calling my function normally, but it's running remotely
    imgs = remote_sd_generate("A hot dog made out of matcha.")
    imgs[0].show()
