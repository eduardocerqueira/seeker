#date: 2025-07-28T16:43:53Z
#url: https://api.github.com/gists/c12f62b0a0de402e39b329d4aea55b1f
#owner: https://api.github.com/users/linoytsaban

!pip install git+https://github.com/huggingface/diffusers.git@refs/pull/12004/head

import torch
import numpy as np
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
dtype = torch.bfloat16
device = "cuda"

pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe.to(device)

# adjust flowshift for phantom LoRA
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=2.0)
pipe.load_lora_weights(
   "vrgamedevgirl84/Wan14BT2VFusioniX", 
   weight_name="FusionX_LoRa/Phantom_Wan_14B_FusionX_LoRA.safetensors", 
    adapter_name="phantom"
)

image = load_image(
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
)
max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))
prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
generator = torch.Generator(device=device).manual_seed(0)
output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=81,
    guidance_scale=3.5,
    num_inference_steps=8, # try between 6-10
    generator=generator,
).frames[0]
export_to_video(output, "i2v_output.mp4", fps=16)