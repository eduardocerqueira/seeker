#date: 2023-06-05T17:09:52Z
#url: https://api.github.com/gists/82dead852e5ed53e90bd5e41be2722b7
#owner: https://api.github.com/users/lumin-leaper

#!/bin/bash -eu
#
# Tested on Ubuntu 20.04
# with 40gig SSD
# 
# Based on: https://webrtc.googlesource.com/src/+/main/docs/native-code/android/index.md
#
# Artifacts will be found in out/ folder.

export PATH="$(pwd)/depot_tools:$PATH"
export WEBRTC_VERSION=4515
export ARCH=x64
export TARGET_OS=android

sudo apt-get install -y pkg-config python-is-python3
git clone --depth 1 https://chromium.googlesource.com/chromium/tools/depot_tools.git
fetch --nohooks webrtc_android
cd src
git checkout "refs/remotes/branch-heads/$WEBRTC_VERSION"
cd ..
gclient sync -f

# Build debug version first
is_debug="true"
output_dir="$(pwd)/out/Debug"
# generate ninja files
gn gen "$output_dir" --root="src" \
  --args="is_debug=${is_debug} \
    target_os=\"$TARGET_OS\" \
    target_cpu=\"$ARCH\" \
    rtc_use_h264=false \
    rtc_include_tests=false \
    rtc_build_examples=false \
    is_component_build=false \
    use_rtti=true \
    use_custom_libcxx=false"
# build static library
ninja -C "$output_dir" webrtc

# Build non debug version
is_debug="false"
output_dir="$(pwd)/out/Release"
# generate ninja files
gn gen "$output_dir" --root="src" \
  --args="is_debug=${is_debug} \
    target_os=\"$TARGET_OS\" \
    target_cpu=\"$ARCH\" \
    rtc_use_h264=false \
    rtc_include_tests=false \
    rtc_build_examples=false \
    is_component_build=false \
    use_rtti=true \
    use_custom_libcxx=false"
# build static library
ninja -C "$output_dir" webrtc