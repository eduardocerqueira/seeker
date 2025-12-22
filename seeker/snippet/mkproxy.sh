#date: 2025-12-22T17:04:33Z
#url: https://api.github.com/gists/623b7b2cb8e428188b1d2f43c279b54a
#owner: https://api.github.com/users/star26bsd

#!/bin/bash
# =============================================================================
# DaVinci Resolve 4:2:2 Proxy Workflow using FFmpeg
# =============================================================================
#
# PROBLEM:
# DaVinci Resolve (even Studio) limits H.265/H.264 proxy generation to 4:2:0
# chroma subsampling. Only ProRes and DNxHR offer 4:2:2 or 4:4:4 — but with
# massive file sizes that defeat the purpose of remote video workflows.
#
# SOLUTION:
# Generate your own HEVC 4:2:2 proxies with ffmpeg, then manually link them
# in Resolve's Media Pool (right-click → Link Proxy Media).
#
# WHY 4:2:2 MATTERS:
# - Color Grading: Smoother gradients, less banding, more push/pull flexibility
# - Chroma Keying: Clean edges on green screen pulls (4:2:0 = blocky artifacts)
# - Remote Workflows: Small files + quality chroma = best of both worlds
#
# REQUIREMENTS:
# - ffmpeg with NVENC support (for GPU acceleration)
# - NVIDIA GPU with NVENC encoder (GTX 1650+, RTX series recommended)
# - For CPU-only encoding, see the libx265 example below
#
# =============================================================================

# -----------------------------------------------------------------------------
# EXAMPLE 1: NVIDIA GPU (NVENC) — 4K to 1080p proxy, 10-bit 4:2:2
# -----------------------------------------------------------------------------
# This is the fastest option if you have an NVIDIA GPU.
# Uses GPU for encoding, CPU for high-quality Lanczos downscaling.

ffmpeg \
  -hwaccel cuda \
  -i "input.mov" \
  -vf "scale=1920:1080:flags=lanczos" \
  -c:v hevc_nvenc \
  -profile:v rext \
  -pix_fmt p210le \
  -preset p5 \
  -cq 28 \
  -rc vbr \
  -lookahead_level 1 \
  -spatial_aq 1 \
  -b_ref_mode 1 \
  -c:a copy \
  "output_proxy.mov"

# KEY PARAMETERS EXPLAINED:
# -hwaccel cuda        → Use NVIDIA CUDA for hardware acceleration
# -profile:v rext      → "Range Extensions" profile, required for 4:2:2/10-bit
# -pix_fmt p210le      → 10-bit 4:2:2 pixel format (little-endian)
# -preset p5           → Balanced speed/quality (p1=fastest, p7=slowest/best)
# -cq 28               → Constant quality (lower=better, 18-28 typical for proxies)
# -rc vbr              → Variable bitrate mode
# -lookahead_level 1   → Enables lookahead for better quality
# -spatial_aq 1        → Adaptive quantization for better detail preservation
# -b_ref_mode 1        → B-frame reference mode for compression efficiency
# -c:a copy            → Copy audio stream without re-encoding


# -----------------------------------------------------------------------------
# EXAMPLE 2: CPU-only (libx265) — For systems without NVIDIA GPU
# -----------------------------------------------------------------------------
# Slower but works on any system. Good for overnight batch processing.

ffmpeg \
  -i "input.mov" \
  -vf "scale=1920:1080:flags=lanczos" \
  -c:v libx265 \
  -profile:v main422-10 \
  -pix_fmt yuv422p10le \
  -preset medium \
  -crf 26 \
  -c:a copy \
  "output_proxy.mov"

# KEY DIFFERENCES:
# -profile:v main422-10  → x265 profile for 4:2:2 10-bit
# -pix_fmt yuv422p10le   → 10-bit 4:2:2 pixel format for software encoding
# -preset medium         → Encoding speed (ultrafast/fast/medium/slow/veryslow)
# -crf 26                → Constant Rate Factor (similar to -cq for NVENC)


# -----------------------------------------------------------------------------
# LINKING PROXIES IN DAVINCI RESOLVE
# -----------------------------------------------------------------------------
# 1. Generate proxies using one of the commands above
# 2. In Resolve, go to Media Pool
# 3. Right-click on the original clip
# 4. Select "Proxy" → "Link Proxy Media"
# 5. Navigate to your generated proxy file
# 6. Resolve will now use the proxy for editing and the original for export
#
# TIP: Keep proxy filenames identical to originals for easier linking!


# -----------------------------------------------------------------------------
# TROUBLESHOOTING
# -----------------------------------------------------------------------------
# "No NVENC capable devices found"
#   → Check NVIDIA drivers, ensure GPU supports NVENC
#
# "Pixel format not supported"
#   → Your GPU may not support 10-bit. Try -pix_fmt p010le (4:2:0 10-bit)
#   → Or fall back to CPU encoding with libx265
#
# "File not recognized in Resolve"
#   → Ensure output container is .mov or .mp4
#   → Check that Resolve version supports HEVC 4:2:2 playback


# =============================================================================
# Author: @star26bsd
# More info: https://x.com/star26bsd
# =============================================================================