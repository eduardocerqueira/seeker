#date: 2026-03-16T17:49:39Z
#url: https://api.github.com/gists/4150ffc99333bcd75ce934d0d922bd33
#owner: https://api.github.com/users/htelsiz

#!/bin/bash
# gaming-setup.sh — run once to set up gaming optimizations on Linux Mint

# GameMode
sudo apt install -y gamemode

# ProtonUp-Qt for GE-Proton
flatpak install -y flathub net.davidotek.pupgui2

# MangoHud for FPS overlay
sudo apt install -y mangohud

# NVIDIA env vars (already done but idempotent)
sudo tee /etc/environment > /dev/null << 'EOF'
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
__NV_PRIME_RENDER_OFFLOAD=1
__VK_LAYER_NV_optimus=NVIDIA_only
__GLX_VENDOR_LIBRARY_NAME=nvidia
EOF

# GameMode config
mkdir -p ~/.config
tee ~/.config/gamemode.ini > /dev/null << 'EOF'
[general]
renice=10

[gpu]
apply_gpu_optimisations=accept-responsibility
gpu_device=0
nv_powermizer_mode=1

[cpu]
park_cores=no
pin_cores=yes
EOF

echo "Done. Reboot to apply."