#date: 2024-05-16T17:00:33Z
#url: https://api.github.com/gists/5c4f1e260a32534732ba9e0321b5b352
#owner: https://api.github.com/users/simbafs

#!/bin/bash

# author: SimbaFs 
# reference: https://github.com/aseprite/aseprite/blob/3ea0437e1dd92b939ff103e5e775397f345cfa52/INSTALL.md

# Define variables
SKIA_DIR="$HOME/deps/skia"
NINJA_URL="https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip"
SKIA_URL="https://github.com/aseprite/skia/releases/download/m102-861e4743af/Skia-Linux-Release-x64-libc++.zip"
ASEPRITE_REPO="https://github.com/aseprite/aseprite.git"
TMP_DIR="/tmp"
ASEPRITE_DIR="$TMP_DIR/aseprite"

# Function to download and extract Skia
download_skia() {
  echo "Downloading pre-built Skia..."
  mkdir -p "$SKIA_DIR"
  cd "$SKIA_DIR" || exit
  wget "$SKIA_URL"
  unzip Skia-Linux-Release-x64-libc++.zip
  rm Skia-Linux-Release-x64-libc++.zip
}

# Check and download Skia if not exists
if [ ! -d "$SKIA_DIR" ]; then
  echo "$SKIA_DIR does not exist, downloading Skia"
  download_skia
else
  echo "$SKIA_DIR exists, skipping download"
fi

# Function to install Ninja
install_ninja() {
  echo "Installing Ninja..."
  cd "$TMP_DIR" || exit
  wget "$NINJA_URL"
  unzip ninja-linux.zip
  sudo mv ninja /usr/local/bin/ninja
  rm ninja-linux.zip
}

# Check and install Ninja if not installed
if command -v ninja >/dev/null 2>&1; then
  echo "Ninja is already installed, skipping"
else
  echo "Ninja is not installed, installing"
  install_ninja
fi

# Clone or update Aseprite repository
if [ ! -d "$ASEPRITE_DIR" ]; then
  echo "Cloning Aseprite repository..."
  cd "$TMP_DIR" || exit
  git clone --depth 1 --recursive "$ASEPRITE_REPO"
  cd "$ASEPRITE_DIR" || exit
  git submodule update --init --recursive
else
  echo "Aseprite repository exists, updating..."
  cd "$ASEPRITE_DIR" || exit
  git pull
  git submodule update --init --recursive
fi

# Install dependencies
echo "Installing dependencies..."
sudo apt-get install -y g++ clang libc++-dev libc++abi-dev cmake ninja-build \
libx11-dev libxcursor-dev libxi-dev libgl1-mesa-dev libfontconfig1-dev

# Build Aseprite
echo "Building Aseprite..."
mkdir -p "$ASEPRITE_DIR/build"
cd "$ASEPRITE_DIR/build" || exit
export CC=clang
export CXX=clang++

cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS:STRING=-stdlib=libc++ \
  -DCMAKE_EXE_LINKER_FLAGS:STRING=-stdlib=libc++ \
  -DLAF_BACKEND=skia \
  -DSKIA_DIR="$SKIA_DIR" \
  -DSKIA_LIBRARY_DIR="$SKIA_DIR/out/Release-x64" \
  -DSKIA_LIBRARY="$SKIA_DIR/out/Release-x64/libskia.a" \
  -DENABLE_WEBP=OFF \
  -G Ninja ..

ninja aseprite

sudo install /tmp/aseprite/build/bin/aseprite /usr/local/bin
sudo mkdir -p /usr/local/share/aseprite
sudo mv data /usr/local/share/aseprite/
