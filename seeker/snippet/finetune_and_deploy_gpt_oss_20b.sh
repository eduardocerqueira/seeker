#date: 2026-03-13T17:14:10Z
#url: https://api.github.com/gists/b2e4598ea82f589655d7ea49e6f30d1d
#owner: https://api.github.com/users/akoumpa

#!/bin/bash

#!/bin/bash
set -euo pipefail

########################################################################
# Fine-tune GPT-OSS 20B with LoRA (PEFT) for 1000 steps, then load
# the saved adapter in vLLM for inference.
#
# Run inside the automodel docker container (transformers v5).
# vLLM inference uses a separate virtual environment (transformers v4).
########################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="openai/gpt-oss-20b"
# you can also pass a local directory
# BASE_MODEL="/lustre/fsw/coreai_dlalgo_llm/akoumparouli/phi4/ckpt/openai/gpt-oss-20b"
TRAIN_CONFIG="examples/llm_finetune/gpt_oss/gpt_oss_20b_single_gpu_peft.yaml"
# running for 100 steps for brevity
MAX_STEPS=100
CKPT_EVERY=${MAX_STEPS}
NUM_GPUS=8

CKPT_DIR="${SCRIPT_DIR}/test_output/gpt_oss_lora"
ADAPTER_DIR="${CKPT_DIR}/epoch_0_step_$((MAX_STEPS-1))/model"
VLLM_VENV="${SCRIPT_DIR}/.vllm_venv"

# =====================================================================
# Step 1 — Fine-tune GPT-OSS with LoRA
# =====================================================================
echo "=== Step 1: Fine-tune ${BASE_MODEL} with LoRA for ${MAX_STEPS} steps ==="
torchrun --nproc-per-node=${NUM_GPUS} examples/llm_finetune/finetune.py \
    --config "${TRAIN_CONFIG}" \
    --model.pretrained_model_name_or_path ${BASE_MODEL} \
    --step_scheduler.max_steps ${MAX_STEPS} \
    --step_scheduler.ckpt_every_steps ${CKPT_EVERY} \
    --step_scheduler.val_every_steps 100 \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir "${CKPT_DIR}" \
    --distributed.ep_size 8 \
    --checkpoint.model_save_format safetensors \
    --peft.target_modules '*_proj' \
    --peft.match_all_linear false \
    --step_scheduler.local_batch_size 1 \
    --step_scheduler.global_batch_size 16

# =====================================================================
# Step 2 — Verify the saved adapter
# =====================================================================
echo ""
echo "=== Step 2: Verify saved adapter ==="
python3 << PYEOF
from safetensors.torch import load_file
import json, sys, os

adapter_dir = "${ADAPTER_DIR}"
st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
cfg_path = os.path.join(adapter_dir, "adapter_config.json")

if not os.path.exists(st_path):
    print(f"ERROR: {st_path} not found")
    sys.exit(1)

state_dict = load_file(st_path)
keys = sorted(state_dict.keys())
print(f"Saved adapter has {len(keys)} keys:")
for k in keys[:10]:
    print(f"  {k}")
if len(keys) > 10:
    print(f"  ... ({len(keys)} total)")

fused = [k for k in keys if "qkv_proj" in k or "gate_up_proj" in k]
split = [k for k in keys if any(p in k for p in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"])]

if fused:
    print(f"\nWARN: Found {len(fused)} fused keys (qkv_proj / gate_up_proj)")
    for k in fused[:5]:
        print(f"  {k}")
else:
    print(f"\nPASS: All {len(split)} keys use split projection names")

with open(cfg_path) as f:
    cfg = json.load(f)
modules = cfg.get("target_modules", [])
print(f"\nadapter_config.json target_modules: {modules}")
fused_modules = [m for m in modules if "qkv_proj" in m or "gate_up_proj" in m]
if fused_modules:
    print(f"WARN: target_modules contains fused names: {fused_modules}")
else:
    print("PASS: target_modules uses split names")
PYEOF

# =====================================================================
# Step 3 — Create vLLM virtual-env (transformers v4)
# =====================================================================
echo ""
echo "=== Step 3: Set up vLLM virtual environment ==="
if [ ! -d "${VLLM_VENV}" ]; then
    echo "Creating venv at ${VLLM_VENV} ..."
    python3 -m venv "${VLLM_VENV}" --system-site-packages
    "${VLLM_VENV}/bin/pip" install --upgrade pip
    "${VLLM_VENV}/bin/pip" install "vllm>=0.8" "transformers>=4,<5"
    echo "vLLM venv ready."
else
    echo "Reusing existing venv at ${VLLM_VENV}"
fi

echo ""
echo "Installed versions in vLLM venv:"
"${VLLM_VENV}/bin/pip" show vllm transformers 2>/dev/null | grep -E "^(Name|Version):"

# =====================================================================
# Step 4 — Load adapter in vLLM and run inference
# =====================================================================
echo ""
echo "=== Step 4: Load adapter with vLLM ==="
VLLM_SCRIPT=$(mktemp /tmp/vllm_gptoss_XXXXXX.py)
trap "rm -f ${VLLM_SCRIPT}" EXIT

cat > "${VLLM_SCRIPT}" << PYEOF
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def main():
    base_model = "${BASE_MODEL}"
    adapter_path = "${ADAPTER_DIR}"

    print(f"Loading base model with vLLM: {base_model}")
    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=64,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    print(f"Running inference with LoRA adapter: {adapter_path}")
    lora_request = LoRARequest("gpt_oss_lora", 1, adapter_path)
    sampling_params = "**********"=64, temperature=0.0)

    prompts = [
        "Hello, how are you?",
        "Explain the theory of relativity in simple terms:",
    ]

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    for out in outputs:
        print(f"\nPrompt : {out.prompt}")
        print(f"Output : {out.outputs[0].text}")

    print("\nPASS: vLLM loaded GPT-OSS LoRA adapter and ran inference successfully")

if __name__ == "__main__":
    main()
PYEOF

"${VLLM_VENV}/bin/python" "${VLLM_SCRIPT}" &
VLLM_PID=$!
wait ${VLLM_PID}
VLLM_EXIT=$?
if [ ${VLLM_EXIT} -ne 0 ]; then
    echo "FAIL: vLLM inference failed with exit code ${VLLM_EXIT}"
    exit 1
fi

echo ""
echo "=== All GPT-OSS LoRA steps passed ==="
 passed ==="
