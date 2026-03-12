#date: 2026-03-12T17:31:32Z
#url: https://api.github.com/gists/f5cd98a708b94f7487229518b6203fc3
#owner: https://api.github.com/users/valientemajp

#!/bin/bash

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI

mkdir -p "${COMFYUI_DIR}/models/checkpoints"
mkdir -p "${COMFYUI_DIR}/models/loras"
mkdir -p "${COMFYUI_DIR}/models/vae"

# SDXL Juggernaut XL Ragnarok (uno de los mejores para NSFW realista)
echo "Descargando Juggernaut XL Ragnarok..."
curl -L -H "Authorization: "**********"
  "https://civitai.com/api/download/models/456194?type=Model&format=SafeTensor" \
  -o "${COMFYUI_DIR}/models/checkpoints/juggernautXL_ragnarok.safetensors"  # ID actual Ragnarok V10 (confirma en Civitai)

if [ $? -eq 0 ]; then
    echo "Juggernaut XL descargado OK."
else
    echo "ERROR en Juggernaut XL."
fi

# VAE para SDXL
curl -L "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors" \
  -o "${COMFYUI_DIR}/models/vae/sdxl_vae.safetensors"

# LoRAs NSFW para SDXL
curl -L -H "Authorization: "**********"
  "https://civitai.com/api/download/models/573152" \
  -o "${COMFYUI_DIR}/models/loras/lustify_sdxl_nsfw.safetensors"  # Lustify GGWP V7 (hardcore)

curl -L "https://civitai.com/api/download/models/1333749" \
  -o "${COMFYUI_DIR}/models/loras/add_detail_xl.safetensors"  # Add Detail XL

curl -L "https://civitai.com/api/download/models/572899" \
  -o "${COMFYUI_DIR}/models/loras/realistic_skin.safetensors"  # Realistic Skin

echo "SDXL provisioning completado."rovisioning completado."