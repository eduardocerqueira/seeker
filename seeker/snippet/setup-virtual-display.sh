#date: 2026-02-12T17:45:47Z
#url: https://api.github.com/gists/9154a619cbf5a5508e5fad336839702b
#owner: https://api.github.com/users/adujardin

#!/bin/bash
# =============================================================================
# Jetson Headless Virtual Display Setup (GPU-Accelerated)
# =============================================================================
#
# Creates a virtual 1920x1080@60Hz display on Jetson AGX Orin without a
# physical monitor, using the NVIDIA driver with full GPU acceleration.
#
# Platform: Jetson AGX Orin
#   - JetPack 5.x (L4T R35.x, Ubuntu 20.04)
#   - JetPack 6.x (L4T R36.x, Ubuntu 22.04)
#
# WHY NOT xserver-xorg-video-dummy?
# ----------------------------------
# The "dummy" driver is software-only — no GPU initialization at boot.
# This means:
#   - No hardware OpenGL/EGL (software rendering only)
#   - NVIDIA Argus (ZED X cameras) crashes on first use because the GPU
#     pipeline isn't initialized, causing a session logout. It only works
#     on the second attempt (after the crash bootstraps the GPU subsystem).
#
# This script instead uses the real nvidia driver with:
#   - ConnectedMonitor: forces the driver to treat a DP output as connected
#   - CustomEDID: provides a fake 1080p monitor description
#   - Result: full GPU init at boot, OpenGL 4.6, Argus works immediately
#
# JP6-SPECIFIC:
#   - Enables nvidia-drm.modeset=1 for proper EGL-CUDA interop
#     (prevents cuGraphicsEGLRegisterImage error 201)
#   - Auto-detects the first available DP connector
#
# USAGE:
#   chmod +x setup-virtual-display.sh
#   sudo ./setup-virtual-display.sh
#   # Then reboot (required, especially on JP6 for modeset)
#
# VERIFY:
#   DISPLAY=:0 xrandr          # Should show DP-x connected 1920x1080
#   DISPLAY=:0 glxinfo | grep "OpenGL renderer"  # Should show Tegra Orin
#
# TO REVERT:
#   sudo cp /etc/X11/xorg.conf.bak.original /etc/X11/xorg.conf
#   sudo rm /etc/X11/edid-1080p.bin
#   sudo systemctl restart gdm3
#   # On JP6 also remove nvidia-drm.modeset=1 from extlinux.conf
#
# =============================================================================

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Must run as root (sudo)."
    exit 1
fi

# --- Detect JetPack / L4T version ---
detect_jetpack_version() {
    local l4t_major=""

    # Try /etc/nv_tegra_release first (JP5 always has this, JP6 may too)
    if [[ -f /etc/nv_tegra_release ]]; then
        l4t_major=$(sed -n 's/^# R\([0-9]*\).*/\1/p' /etc/nv_tegra_release)
    fi

    # Fallback: check nvidia-l4t-core package version
    if [[ -z "$l4t_major" ]] && command -v dpkg-query &>/dev/null; then
        l4t_major=$(dpkg-query -W -f='${Version}' nvidia-l4t-core 2>/dev/null \
                    | sed -n 's/^\([0-9]*\)\..*/\1/p') || true
    fi

    # Map L4T major to JetPack
    if [[ -n "$l4t_major" && "$l4t_major" -ge 36 ]]; then
        echo 6
    elif [[ -n "$l4t_major" && "$l4t_major" -ge 35 ]]; then
        echo 5
    else
        echo 0  # unknown
    fi
}

# --- Detect first available DP connector ---
detect_dp_connector() {
    local connector=""

    # Check /sys/class/drm for DP connectors (card0-DP-1 → DP-0, card1-DP-2 → DP-1, etc.)
    for card_dp in /sys/class/drm/card*-DP-*; do
        if [[ -d "$card_dp" ]]; then
            # Extract connector name: card0-DP-1 → DP-0 (Xorg uses 0-indexed)
            local raw_name
            raw_name=$(basename "$card_dp")
            # DRM uses 1-indexed (DP-1), Xorg uses 0-indexed (DP-0)
            local drm_idx
            drm_idx=$(echo "$raw_name" | sed 's/.*DP-//')
            connector="DP-$((drm_idx - 1))"
            break
        fi
    done

    # Fallback
    if [[ -z "$connector" ]]; then
        connector="DP-0"
    fi

    echo "$connector"
}

JP_VERSION=$(detect_jetpack_version)
DP_CONNECTOR=$(detect_dp_connector)

echo "=== Jetson Virtual Display Setup ==="

if [[ "$JP_VERSION" -eq 6 ]]; then
    echo "Detected: JetPack 6 (L4T R36.x)"
elif [[ "$JP_VERSION" -eq 5 ]]; then
    echo "Detected: JetPack 5 (L4T R35.x)"
else
    echo "WARNING: Could not detect JetPack version, proceeding with defaults"
fi
echo "Using display connector: ${DP_CONNECTOR}"

TOTAL_STEPS=4
[[ "$JP_VERSION" -eq 6 ]] && TOTAL_STEPS=5

# --- Step 1: Backup existing xorg.conf ---
if [[ -f /etc/X11/xorg.conf ]]; then
    if [[ ! -f /etc/X11/xorg.conf.bak.original ]]; then
        cp /etc/X11/xorg.conf /etc/X11/xorg.conf.bak.original
        echo "[1/${TOTAL_STEPS}] Backed up original xorg.conf"
    else
        echo "[1/${TOTAL_STEPS}] Original backup already exists, skipping"
    fi
else
    echo "[1/${TOTAL_STEPS}] No existing xorg.conf to back up"
fi

# --- Step 2: Install 1080p EDID binary ---
# Valid 128-byte EDID block describing a 1920x1080@60Hz monitor.
# Fed to the nvidia driver via CustomEDID so it believes a real monitor
# is connected.
base64 -d > /etc/X11/edid-1080p.bin << 'EDID_EOF'
AP///////wBZ5QEAAQAAAAEiAQOAAAB4Cu6Ro1RMmSYPUFQhCAABAQEBAQEBAQEBAQEBAQEBAjqA
GHE4LUBYLEUADyghAAAeAAAA/ABWaXJ0dWFsIDEwODBwAAAA/QAySx5REQAKICAgICAgAAAA/wAw
MDAwMDAwMDAwMDAxAJc=
EDID_EOF
chmod 644 /etc/X11/edid-1080p.bin
echo "[2/${TOTAL_STEPS}] Installed EDID binary to /etc/X11/edid-1080p.bin"

# --- Step 3: Write xorg.conf ---
cat > /etc/X11/xorg.conf << XORG_EOF
# Jetson AGX Orin - Virtual display with full NVIDIA GPU acceleration
# No physical monitor required. Uses ConnectedMonitor + custom EDID
# to force the nvidia driver to create a proper GPU-accelerated screen.
# Generated for JetPack ${JP_VERSION:-unknown}, connector ${DP_CONNECTOR}

Section "DRI"
    Mode 0666
EndSection

Section "Module"
    Disable     "dri"
    SubSection  "extmod"
        Option  "omit xfree86-dga"
    EndSubSection
EndSection

Section "Device"
    Identifier  "Tegra0"
    Driver      "nvidia"

    # Force the driver to believe a DP monitor is connected
    Option      "AllowEmptyInitialConfiguration" "true"
    Option      "ConnectedMonitor"               "${DP_CONNECTOR}"
    Option      "UseDisplayDevice"               "${DP_CONNECTOR}"

    # Use our custom EDID so the driver knows the resolution
    Option      "CustomEDID"                     "${DP_CONNECTOR}:/etc/X11/edid-1080p.bin"

    # Disable power management that can invalidate GPU contexts
    Option      "HardDPMS"                       "false"
EndSection

Section "Monitor"
    Identifier  "Virtual-1080p"
    HorizSync   30-81
    VertRefresh 50-75
    ModeLine    "1920x1080_60" 148.50 1920 2008 2052 2200 1080 1084 1089 1125 +HSync +VSync
    Option      "PreferredMode" "1920x1080_60"
EndSection

Section "Screen"
    Identifier  "Default Screen"
    Device      "Tegra0"
    Monitor     "Virtual-1080p"
    DefaultDepth 24
    SubSection "Display"
        Depth    24
        Modes    "1920x1080_60"
    EndSubSection
EndSection
XORG_EOF
echo "[3/${TOTAL_STEPS}] Wrote /etc/X11/xorg.conf"

# --- Step 4: Ensure nvidia-drm module loads at boot ---
# The nvidia-drm module creates /dev/dri/card0 and /dev/dri/renderD128,
# which are required for EGL to expose EGL_EXT_device_drm and for
# cuGraphicsEGLRegisterImage CUDA-EGL interop to work.
# Without this, the ZED SDK's GPU buffer registration fails with error 201.
# The module MUST be loaded before X/GDM starts.
MODULES_CONF="/etc/modules-load.d/nvidia-drm.conf"
if [[ -f "$MODULES_CONF" ]] && grep -q "nvidia-drm" "$MODULES_CONF"; then
    echo "[4/${TOTAL_STEPS}] nvidia-drm already in ${MODULES_CONF}"
else
    echo "nvidia-drm" >> "$MODULES_CONF"
    echo "[4/${TOTAL_STEPS}] Added nvidia-drm to ${MODULES_CONF} (loads at boot before X)"
fi

# --- Step 5 (JP6 only): Enable nvidia-drm.modeset=1 ---
# Required on JP6 for proper DRM/KMS initialization, which in turn
# ensures EGL-CUDA interop works. JP5 doesn't need modeset.
if [[ "$JP_VERSION" -eq 6 ]]; then
    EXTLINUX_CONF="/boot/extlinux/extlinux.conf"
    if [[ -f "$EXTLINUX_CONF" ]]; then
        if grep -q "nvidia-drm.modeset=1" "$EXTLINUX_CONF"; then
            echo "[5/${TOTAL_STEPS}] nvidia-drm.modeset=1 already set in extlinux.conf"
        else
            # Backup extlinux.conf
            cp "$EXTLINUX_CONF" "${EXTLINUX_CONF}.bak.$(date +%Y%m%d%H%M%S)"
            # Append to the APPEND line
            sed -i '/^\s*APPEND /s/$/ nvidia-drm.modeset=1/' "$EXTLINUX_CONF"
            echo "[5/${TOTAL_STEPS}] Added nvidia-drm.modeset=1 to ${EXTLINUX_CONF}"
        fi
    else
        echo "[5/${TOTAL_STEPS}] WARNING: ${EXTLINUX_CONF} not found."
        echo "         Manually add 'nvidia-drm.modeset=1' to your kernel boot params."
    fi
fi

# --- Detect display manager ---
DM="gdm3"
if systemctl cat gdm.service &>/dev/null && \
   ! systemctl cat gdm3.service &>/dev/null; then
    DM="gdm"
fi

echo ""
echo "=== Done! ==="
if [[ "$JP_VERSION" -eq 6 ]]; then
    echo "A full reboot is REQUIRED for nvidia-drm.modeset=1 to take effect:"
    echo "  sudo reboot"
else
    echo "Restart the display manager to apply:"
    echo "  sudo systemctl restart ${DM}"
    echo ""
    echo "Or reboot the system."
fi
echo ""
echo "Verify with:"
echo "  DISPLAY=:0 xrandr"
echo "  DISPLAY=:0 glxinfo | grep 'OpenGL renderer'"
