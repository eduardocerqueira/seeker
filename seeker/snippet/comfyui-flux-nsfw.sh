#date: 2026-03-12T17:31:02Z
#url: https://api.github.com/gists/00c7faec5e1ccbf6bb63be5cfa9a435c
#owner: https://api.github.com/users/valientemajp

#!/bin/bash

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI

mkdir -p "${COMFYUI_DIR}/models/checkpoints"
mkdir -p "${COMFYUI_DIR}/models/loras"
mkdir -p "${COMFYUI_DIR}/models/vae"

echo "Descargando Flux.1 dev fp16..."
curl -L -H "Authorization: "**********"
  "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors" \
  -o "${COMFYUI_DIR}/models/checkpoints/flux1-dev.safetensors"

if [ $? -eq 0 ]; then
    echo "Flux.1 dev descargado OK."
else
    echo "ERROR en Flux.1 dev. Verifica HF_TOKEN o descarga manual."
fi

# LoRAs NSFW recomendados para Flux (realismo explícito)
curl -L "https://civitai.com/api/download/models/[ID_FLUX_NSFW_1]" \
  -o "${COMFYUI_DIR}/models/loras/flux_nsfw_anatomy.safetensors"  # Sustituye ID (busca "Flux NSFW anatomy" en Civitai)

curl -L "https://civitai.com/api/download/models/[ID_FLUX_DETAIL]" \
  -o "${COMFYUI_DIR}/models/loras/flux_detail_enhancer.safetensors"  # Detalle piel y anatomía

echo "Flux provisioning completado."etado."