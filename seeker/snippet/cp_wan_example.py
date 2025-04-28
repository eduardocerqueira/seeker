#date: 2025-04-28T17:01:57Z
#url: https://api.github.com/gists/93b467ddf64bfe9df47fc12fc2ae4fac
#owner: https://api.github.com/users/a-r-r-o-w

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

from finetrainers.parallel.ptd import apply_cp
from finetrainers._metadata import ParamId, CPInput, CPOutput
from finetrainers.models.attention_dispatch import attention_provider, attention_dispatch

torch.nn.functional.scaled_dot_product_attention = attention_dispatch

dist.init_process_group("nccl")
rank, world_size = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
cp_mesh = dist.device_mesh.init_device_mesh("cuda", [world_size], mesh_dim_names=["cp"])

cp_plan = {
    "blocks.*": {
        ParamId("encoder_hidden_states", 1): CPInput(split_dim=1, expected_dims=3),
        ParamId("rotary_emb", 3): CPInput(split_dim=2, expected_dims=4),
    },
    "blocks.0": {ParamId("hidden_states", 0): CPInput(split_dim=1, expected_dims=3)},
    "proj_out": [CPOutput(gather_dim=1, expected_dims=3)],
}

try:
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    apply_cp(pipe.transformer, mesh=cp_mesh, plan=cp_plan)

    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

    prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt, device="cuda",
        )

    with attention_provider("sage", mesh=cp_mesh, convert_to_fp32=True, rotate_method="allgather"):
        latents = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=5.0,
            output_type="latent"
        ).frames[0]
    
    with torch.no_grad():
        latents = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = pipe.vae.decode(latents, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
    export_to_video(video, "output.mp4", fps=16)
finally:
    dist.destroy_process_group()
