#date: 2025-04-21T16:53:59Z
#url: https://api.github.com/gists/d4cfc65d81df5baf1d6f433c8506629f
#owner: https://api.github.com/users/brooklyn097

#!/bin/bash
# =========================================================================
# Source: https://gist.github.com/lucaspar/27f5e108b80524b315be10b2a9049817
# =========================================================================
# This script will compile and install a static FFmpeg build with
#   support for NVENC in Ubuntu. Updated for Ubuntu 24.04.2,
#   with NVIDIA Drivers v535.183.01 and CUDA v12.2 with a GPU
#   with CUDA capability 8.6 (RTX 3080). Set ccap below if using
#   a different GPU.
# It assumes NVIDIA drivers are installed and that you have a
#   CUDA-compatible GPU. You can check installed drivers with:
#       $ apt list *nvidia-driver-* | grep installed
#       $ nvidia-smi
# The script may be run multiple times if a step fails.
# =========================================================================
set -e

# Variables you might want to change
DIR_USR_BIN="${HOME}/.local/bin"                           # user-writable binaries, where to install ffmpeg
DIR_INSTALL_ROOT="${XDG_STATE_HOME:-${HOME}/.local/state}" # location to clone repos and build artifacts
DIR_FFMPEG_BUILD="${DIR_INSTALL_ROOT}/ffmpeg-build"        # where to build ffmpeg
DIR_FFMPEG_SOURCES="${DIR_INSTALL_ROOT}/ffmpeg-sources"    # ffmpeg source code

# CUDA compute capability: check yours at https://developer.nvidia.com/cuda-gpus, or run:
#   nvidia-smi --query-gpu=compute_cap --format=csv
ccap=86

# NASM (Netwide Assembler) version to install (most recent version at the time of writing)
# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#NASM
NASM_VERSION=2.16.01

# ffmpeg version to install (most recent version at the time of writing)
#   check versions at https://github.com/FFmpeg/FFmpeg/branches/active
#   check changelog at https://github.com/FFmpeg/FFmpeg/blob/master/Changelog
FFMPEG_VERSION=7.1

# Install required things from apt
function install_libs() {
    echo " 🚀 Installing prerequisites"
    sudo apt-get update
    sudo apt-get -y install autoconf automake build-essential \
        libass-dev libfreetype6-dev libsdl1.2-dev libtheora-dev \
        libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev \
        libxcb-xfixes0-dev pkg-config texi2html zlib1g-dev libopus-dev libunistring-dev \
        libavdevice-dev libfdk-aac-dev libmp3lame-dev libx264-dev libavcodec-dev \
        libgnutls28-dev libx265-dev libnuma-dev libaom-dev libaom3
    # install cuda toolkit, which provides nvcc
    sudo apt-get -y install nvidia-cuda-toolkit
}

# Install NVENC SDK
function install_nvenc_sdk() {
    echo " 🚀 Installing the NVIDIA NVENC SDK."
    cd "${DIR_FFMPEG_SOURCES}" || exit 1
    dir_nvcodec="${DIR_FFMPEG_SOURCES}/nv-codec-headers"
    if [ ! -d "${dir_nvcodec}" ]; then
        git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git "${dir_nvcodec}"
    fi
    cd "${dir_nvcodec}" || exit 1
    make
    sudo make install
}

# Compile NASM
function compile_nasm() {
    echo " 🚀 Compiling nasm"
    dir_nasm="${DIR_FFMPEG_SOURCES}/nasm-${NASM_VERSION}"
    if [ ! -d "${dir_nasm}" ]; then
        cd "${DIR_FFMPEG_SOURCES}" || exit 1
        wget "https://www.nasm.us/pub/nasm/releasebuilds/${NASM_VERSION}/nasm-${NASM_VERSION}.tar.gz"
        tar xzvf nasm-${NASM_VERSION}.tar.gz && rm -f nasm-${NASM_VERSION}.tar.gz
    fi
    cd "${dir_nasm}" || exit 1
    ./configure --prefix="${DIR_FFMPEG_BUILD}" --bindir="${DIR_USR_BIN}"
    make -j"${NUM_CORES}" || true         # too many false positives, ignoring errors
    make -j"${NUM_CORES}" install || true # too many false positives, ignoring errors
    make -j"${NUM_CORES}" distclean
}

# Compile libvpx
function compile_libvpx() {
    echo " 🚀 Compiling libvpx"
    dir_vpx="${DIR_FFMPEG_SOURCES}/libvpx"
    if [ ! -d "${dir_vpx}" ]; then
        cd "${DIR_FFMPEG_SOURCES}" || exit 1
        git clone https://chromium.googlesource.com/webm/libvpx "${dir_vpx}"
    fi
    cd "${dir_vpx}" || exit 1
    PATH="${DIR_USR_BIN}:${PATH}" ./configure --prefix="${DIR_FFMPEG_BUILD}" --disable-examples \
        --enable-runtime-cpu-detect --enable-vp9 --enable-vp8 \
        --enable-postproc --enable-vp9-postproc --enable-multi-res-encoding \
        --enable-webm-io --enable-better-hw-compatibility --enable-vp9-highbitdepth \
        --enable-onthefly-bitpacking --enable-realtime-only --enable-pic \
        --cpu=native --as=nasm
    PATH="${DIR_USR_BIN}:${PATH}" make -j"${NUM_CORES}"
    make -j"${NUM_CORES}" install
    make -j"${NUM_CORES}" clean
}

# Compile ffmpeg and install it
function compile_and_install_ffmpeg() {
    echo " 🚀 Compiling ffmpeg"
    dir_ffmpeg="${DIR_FFMPEG_SOURCES}/ffmpeg-repo"
    if [ ! -d "${dir_ffmpeg}" ]; then
        cd "${DIR_FFMPEG_SOURCES}" || exit 1
        git clone https://github.com/FFmpeg/FFmpeg.git -b master "${dir_ffmpeg}"
    fi
    cd "${dir_ffmpeg}" || exit 1
    git fetch --tags
    git switch "release/${FFMPEG_VERSION}"
    PATH="${DIR_USR_BIN}:${PATH}" PKG_CONFIG_PATH="${DIR_FFMPEG_BUILD}/lib/pkgconfig" ./configure \
        --pkg-config-flags="--static" \
        --prefix="${DIR_FFMPEG_BUILD}" \
        --extra-cflags="-I${DIR_FFMPEG_BUILD}/include" \
        --extra-ldflags="-L${DIR_FFMPEG_BUILD}/lib" \
        --extra-cflags="-I/usr/local/cuda/include/" \
        --extra-ldflags=-L/usr/local/cuda/lib64/ \
        --nvccflags="-gencode arch=compute_${ccap},code=sm_${ccap} -O2" \
        --bindir="${DIR_USR_BIN}" \
        --enable-static \
        --enable-cuda-nvcc \
        --enable-cuvid \
        --enable-decoder=aac \
        --enable-decoder=h264 \
        --enable-decoder=h264_cuvid \
        --enable-demuxer=mov \
        --enable-filter=scale \
        --enable-gnutls \
        --enable-gpl \
        --enable-libass \
        --enable-libfdk-aac \
        --enable-libfreetype \
        --enable-libmp3lame \
        --enable-libnpp \
        --enable-libopus \
        --enable-libtheora \
        --enable-libvorbis \
        --enable-libvpx \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libaom \
        --enable-nonfree \
        --enable-nvdec \
        --enable-nvenc \
        --enable-pic \
        --enable-protocol=file \
        --enable-protocol=https \
        --enable-vaapi
    PATH="${DIR_USR_BIN}:${PATH}" make -j"${NUM_CORES}"
    make -j"${NUM_CORES}" install
    make -j"${NUM_CORES}" distclean
    hash -r
}

# check if nvidia-smi succeeds
function check_nvidia_smi() {
    if ! nvidia-smi &>/dev/null; then
        echo -e "\tnvidia-smi not found or failed to run."
        echo -e "\tMake sure NVIDIA drivers are installed e.g.:"
        echo -e "\t\tsudo apt install nvidia-driver-545"
        exit 1
    fi
}

# link installed binaries to user bin
function link_installed() {
    echo "Linking installed binaries to ${DIR_USR_BIN}"
    cd "${DIR_USR_BIN}" || exit 1
    declare -a arr=("ffmpeg" "ffprobe" "ffplay" "ffserver" "ffprobe")
    for f in "${arr[@]}"; do
        f_path="${DIR_FFMPEG_BUILD}/bin/${f}"
        if [ -f "${f_path}" ]; then
            ldd "${f_path}"
        fi
    done
}

# main things
function main() {

    check_nvidia_smi

    # set number of cores to use
    num_proc="$(nproc)"
    NUM_CORES=$((num_proc - 2))
    echo "Using up to ${NUM_CORES} cores"

    # create directories
    DIR_PREV="$(pwd)"
    cd "${DIR_INSTALL_ROOT}" || exit 1
    mkdir -p "${DIR_FFMPEG_SOURCES}"

    install_libs
    install_nvenc_sdk
    compile_nasm
    compile_libvpx
    compile_and_install_ffmpeg

    cd "${DIR_PREV}" || exit 1
    link_installed

    wrap_up
    benchmark

}

function wrap_up() {

    echo " 🏁 Complete!"

    echo -e " 💡 Some commands to try:\n"
    echo "which ffmpeg ffprobe"
    echo "whereis ffmpeg ffprobe    # to run the correct binary in case your system has others"
    echo "ffmpeg 2>/dev/null -version | grep -i 'nv[a-z]*|nvidia|cuda|nvenc'"
    echo "ffprobe 2>/dev/null -decoders | grep -i 'nvidia|cuda|nvenc'"
    echo "ffprobe 2>/dev/null -encoders | grep -i 'nvidia|cuda|nvenc'"
    echo "ffprobe 2>/dev/null -filters | grep -i 'nvidia|cuda|nvenc'"

    echo -e "\n\n\t\t EXAMPLE OF NVENC USAGE"
    echo -e "\n 👉 Get a sample video to work with:\n"
    echo -e "wget https://delamain.s3.amazonaws.com/public/samples/video-sample-720p.mp4 -O input.mp4"
    echo -e "\n 👉 GPU encoding test:\n"
    echo -e "time ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -extra_hw_frames 4 -i input.mp4 -c:v h264_nvenc output-nvenc.mp4"
    echo -e "\n 👉 CPU encoding test:\n"
    echo -e "time ffmpeg -y -i input.mp4 -c:v h264 output-cpu.mp4"

    echo -e "\n 👉 Wall times for tested run:\t15.695s (CPU, 24 cores) vs. (CUDA, RTX 3080) 4.952s"

    echo -e "\nIf everything works, you can mark the following directories for deletion:\n\n 🧹 ${DIR_FFMPEG_SOURCES}\n 🧹 ${DIR_FFMPEG_BUILD}\n"

}

function benchmark() {

    if ! [ -f input.mp4 ]; then
        echo "Downloading sample video"
        wget https://delamain.s3.amazonaws.com/public/samples/video-sample-720p.mp4 -O input.mp4
    fi

    echo "Benchmarking x264 (MPEG-4/H.264) CPU vs. NVENC"
    time ffmpeg -y -i input.mp4 -c:v h264 output-cpu-h264.mp4
    time ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -extra_hw_frames 4 -i input.mp4 -c:v h264_nvenc output-nvenc-h264.mp4

    echo "Benchmarking x265 (HEVC/H.265) CPU vs. NVENC"
    time ffmpeg -y -i input.mp4 -c:v hevc output-cpu-hevc.mp4
    time ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -extra_hw_frames 4 -i input.mp4 -c:v hevc_nvenc output-nvenc-hevc.mp4

}

main
