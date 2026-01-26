#date: 2026-01-26T17:06:46Z
#url: https://api.github.com/gists/4479867a4cd4d639df42eea28b9f4ff6
#owner: https://api.github.com/users/tin2tin

import time
import torch
import json
import os
import gc
import numpy as np
from huggingface_hub import hf_hub_download
from diffusers import GGUFQuantizationConfig, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.utils import load_image

# -------------------------------------------------------------------------
# START TIMER
# -------------------------------------------------------------------------
start_time = time.time()
print("Script started...")

# -------------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------------
device = "cuda"
width = 768
height = 512
random_seed = 45
generator = torch.Generator(device).manual_seed(random_seed)

# Paths
gguf_ckpt = "https://huggingface.co/unsloth/LTX-2-GGUF/blob/main/ltx-2-19b-dev-Q4_K_M.gguf"
output_path = r"C:\Users\peter\Downloads\image_ltx2_distilled_sample.mp4"
config_model_id = "rootonchair/LTX-2-19b-distilled" 

# ---------------------------------------------------------
# 1. Patch & Save Configuration
# ---------------------------------------------------------
local_config_dir = "./ltx_patched_config"
os.makedirs(local_config_dir, exist_ok=True)

print(f"Downloading config from {config_model_id}...")
config_path = hf_hub_download(config_model_id, "config.json", subfolder="transformer")

with open(config_path, "r") as f:
    config_dict = json.load(f)

# Patching for Dev weights compatibility (7680 channels vs 3840 in distilled config)
config_dict["caption_channels"] = 7680
local_config_path = os.path.join(local_config_dir, "config.json")
with open(local_config_path, "w") as f:
    json.dump(config_dict, f)

# ---------------------------------------------------------
# 2. Load Models
# ---------------------------------------------------------
print("Loading Models...")
quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

transformer = LTX2VideoTransformer3DModel.from_single_file(
    gguf_ckpt,
    config=local_config_dir, 
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)

pipe = LTX2ImageToVideoPipeline.from_pretrained(
    config_model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16
)
# Use model offload to handle Transformer memory automatically
pipe.enable_model_cpu_offload()

# OPTIMIZATION: Enable VAE Tiling/Slicing to reduce decode VRAM
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# ---------------------------------------------------------
# 3. Stage 1: Base Generation
# ---------------------------------------------------------
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg")
prompt = "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a breath-taking, movie-like shot."
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

print("Generating Stage 1 latents...")
video_latent, audio_latent = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=24.0,
    num_inference_steps=8,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0, 
    generator=generator,
    output_type="latent",
    return_dict=False,
)

# ---------------------------------------------------------
# 4. Stage 2: Upsampling
# ---------------------------------------------------------
print("Loading Upsampler...")
latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    config_model_id,
    subfolder="latent_upsampler",
    torch_dtype=torch.bfloat16,
)

# Crucial: pass pipe.vae (existing instance) to the upsampler pipeline
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload()

print("Upsampling latents...")
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

# Cleanup Upsampler to free VRAM for the final pass
del upsample_pipe
del latent_upsampler
gc.collect()
torch.cuda.empty_cache()

# ---------------------------------------------------------
# 5. Stage 3: Refinement (Output Latents to save VRAM)
# ---------------------------------------------------------
print("Refining upscaled latents...")

# We use the original pipe again (Transformer will load back to GPU automatically via hooks)
# IMPORTANT: We use output_type="latent" to skip the massive single-batch decode
refined_latents, audio = pipe(
    image=image,
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width * 2,
    height=height * 2,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    generator=generator,
    guidance_scale=1.0,
    output_type="latent", # <--- Returns latents, avoids 39GB decode spike
    return_dict=False,
)

# ---------------------------------------------------------
# 6. Manual Chunked Decoding
# ---------------------------------------------------------
print("Decoding video in chunks...")

# Ensure VAE is ready
vae = pipe.vae

# Scale latents according to LTX config (Inverse of encoding)
refined_latents = refined_latents / vae.config.scaling_factor

decoded_frames_list = []
batch_size = 4 # Decode 4 frames at a time. Safe for <24GB VRAM.
total_frames = refined_latents.shape[2]

# Move latents to the same device as VAE is expected to be
refined_latents = refined_latents.to(device, dtype=torch.bfloat16)

with torch.no_grad():
    for i in range(0, total_frames, batch_size):
        print(f"Decoding frames {i} to {min(i+batch_size, total_frames)}...")
        
        # Slice: [Batch, Channels, Frames, Height, Width]
        latent_chunk = refined_latents[:, :, i:i+batch_size, :, :]
        
        # Decode using the existing VAE instance
        decoded_chunk = vae.decode(latent_chunk, return_dict=False)[0]
        
        # Post-process: [-1, 1] -> [0, 1]
        decoded_chunk = (decoded_chunk / 2 + 0.5).clamp(0, 1)
        
        # Move to CPU immediately
        decoded_frames_list.append(decoded_chunk.cpu())
        
        torch.cuda.empty_cache()

print("Stitching frames...")
# Concatenate all chunks
video = torch.cat(decoded_frames_list, dim=2)
# Convert to uint8 (0-255)
video = (video * 255).byte()
# Permute to [Batch, Frames, Height, Width, Channels]
video = video.permute(0, 2, 3, 4, 1)

# ---------------------------------------------------------
# 7. Export
# ---------------------------------------------------------
print(f"Saving to {output_path}...")

# FIX: Remove 'audio' and 'audio_sample_rate' arguments.
# 'audio' contains raw latents, not a waveform, and we deleted the vocoder.
encode_video(
    video[0],
    fps=24.0,
    output_path=output_path,
)

# ---------------------------------------------------------
# END TIMER
# ---------------------------------------------------------
end_time = time.time()
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60

print("---------------------------------------------------------")
print(f"Done! Video saved to {output_path}")
print(f"Total Execution Time: {elapsed_minutes:.2f} minutes ({elapsed_seconds:.0f} seconds)")
print("---------------------------------------------------------")