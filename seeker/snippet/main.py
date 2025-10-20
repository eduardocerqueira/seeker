#date: 2025-10-20T17:10:20Z
#url: https://api.github.com/gists/a3be9d75ab0965ebada24308fdeb792e
#owner: https://api.github.com/users/ustas-eth

from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageOps
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "deepseek-ai/DeepSeek-OCR"

# --- Load model/tokenizer (no flash attention) ---
tokenizer = "**********"=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation="eager",
    trust_remote_code=True,
    use_safetensors=True,
)
# Cast on CPU first, then move to GPU (avoids fp32 staging on GPU)
model = model.eval().to(torch.bfloat16).cuda()

# --- Force a higher temperature for infer() without editing the repo ---
orig_generate = model.generate
def _generate_with_temp(*args, **kwargs):
    kwargs["do_sample"] = True          # enable sampling
    kwargs["temperature"] = 0.2         # pick your value, e.g. 0.6–0.7
    kwargs.setdefault("top_p", 0.95)    # optional but common
    kwargs.pop("num_beams", None)       # ensure beam params don't conflict
    return orig_generate(*args, **kwargs)

model.generate = _generate_with_temp

# --- Inputs ---
img_a = "Untitled-2025-10-20-1610-small.png"
img_b = "Untitled-2025-10-20-1610-small-2.png"
output_path = "output"
os.makedirs(output_path, exist_ok=True)

# --- Make a single side-by-side image so infer() can accept it ---
def open_exif_safe(path: str) -> Image.Image:
    img = Image.open(path)
    return ImageOps.exif_transpose(img).convert("RGB")

A = open_exif_safe(img_a)
B = open_exif_safe(img_b)

# Normalize heights by padding the shorter one
h = max(A.height, B.height)
if A.height < h:
    padded = Image.new("RGB", (A.width, h), (255, 255, 255))
    padded.paste(A, (0, 0))
    A = padded
if B.height < h:
    padded = Image.new("RGB", (B.width, h), (255, 255, 255))
    padded.paste(B, (0, 0))
    B = padded

# Optional: small vertical divider
divider_w = 8
divider = Image.new("RGB", (divider_w, h), (255, 255, 255))

combined = Image.new("RGB", (A.width + divider_w + B.width, h), (255, 255, 255))
combined.paste(A, (0, 0))
combined.paste(divider, (A.width, 0))
combined.paste(B, (A.width + divider_w, 0))

combined_path = os.path.join(output_path, "combined_side_by_side.png")
combined.save(combined_path)

# --- Ask the model about the two diagrams ---
prompt = "<image>\nWhat's the difference between these two diagrams?"
# NOTE: Left side is the first image; right side is the second.

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=combined_path,   # single path (combined image)
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True,
)
  save_results=True,
    test_compress=True,
)
