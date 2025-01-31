#date: 2025-01-31T16:40:06Z
#url: https://api.github.com/gists/3bef6bf38cb86b5e1e4cdefcfc006f7e
#owner: https://api.github.com/users/devkittencpp

#!/bin/bash

# Check current GCC and G++ versions
current_gcc=$(gcc -dumpversion 2>/dev/null)
current_gpp=$(g++ -dumpversion 2>/dev/null)

echo "Current GCC version: $current_gcc"
echo "Current G++ version: $current_gpp"

# Desired version
required_version="13"

# Update package lists and enable necessary repositories
echo "Updating system and enabling necessary repositories..."
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager --set-enabled updates-testing

# List of required packages
packages=(
    bzip2
    git
    make
    autoconf
    cmake
    mesa-libGL-devel
    boost-devel
    zlib-devel
    wget
    xorg-x11-server-Xvfb
    flex
    bison
    libXcursor-devel
    libXcomposite-devel
    gcc-c++
    openssl-devel
    libxcb-devel
    libX11-devel
    systemd-devel
    SDL2-devel
    qt5-qtbase-devel
    qt5-qtdeclarative-devel
    qt5-qttools-devel
    qt5-qtx11extras-devel
    qt5-qtmultimedia-devel
    freeglut-devel
    bzip2-devel
    lua-devel
    gcc-toolset-13-gcc
    gcc-toolset-13-gcc-c++
)

# Install packages
echo "Installing required packages..."
sudo dnf install -y "${packages[@]}"

# Clone and install StormLib
echo "Cloning and installing StormLib..."
if git clone https://github.com/ladislav-zezula/StormLib.git; then
    cd StormLib
    cmake .
    make -j"$(nproc)"
    sudo make install
    cd ..
fi

# Clone and install LuaJIT
echo "Cloning and installing LuaJIT..."
if git clone https://github.com/LuaJIT/LuaJIT.git; then
    cd LuaJIT
    make -j"$(nproc)"
    sudo make install
    cd ..
fi

# Function to enable GCC 13
update_compiler() {
    echo "Enabling GCC 13 from Software Collection (SCL)..."
    source /opt/rh/gcc-toolset-13/enable
    echo "GCC and G++ have been switched to version $required_version."
}

# Check if the versions are already correct
if [[ "$current_gcc" == "$required_version" && "$current_gpp" == "$required_version" ]]; then
    echo "GCC and G++ are already set to version $required_version."
else
    echo "Updating GCC and G++ to version $required_version..."
    update_compiler
fi

# Verify changes
echo "New GCC version: $(gcc -dumpversion)"
echo "New G++ version: $(g++ -dumpversion)"
