#date: 2025-04-14T17:03:53Z
#url: https://api.github.com/gists/192770a88e74794d18914d46bad4540b
#owner: https://api.github.com/users/dk1aj

#!/bin/bash
# -----------------------------------------------------------------------------
# Copyleft (?) 14.04.2025 DK1AJ <dk1aj@dg-email.de>
# This script is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This script is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# -----------------------------------------------------------------------------

set -e  # Exit script on any error

# Define log file
LOG_FILE="$HOME/root_install.log"
exec > >(tee -a "$LOG_FILE") 2>&1  # Log all output to file

echo "###-START-###"
echo "Checking environment..."

# --- Check if running on a Raspberry Pi ---
if ! grep -q 'Raspberry' /proc/cpuinfo; then
    echo "WARNING: This script is intended for Raspberry Pi systems only!"
    echo "Proceeding anyway..."
fi

# --- Synchronize system time ---
echo "Synchronizing system time via NTP..."
sudo timedatectl set-ntp true

echo "Preparing system with required packages..."

# --- Update package lists ---
sudo apt update -y

# Define required packages
PACKAGES=(
    mc g++ cmake make libsigc++-2.0-dev libgsm1-dev libpopt-dev tcl tcl-dev
    libgcrypt20-dev libspeex-dev libasound2-dev libopus-dev librtlsdr-dev
    doxygen groff alsa-utils vorbis-tools curl libvorbis-dev git bc rtl-sdr
    libcurl4-openssl-dev libjsoncpp-dev libgpiod2 libgpiod-dev gpiod
    build-essential gcc gzip tar graphviz libsigc++-2.0-dev libspeexdsp-dev
    libogg-dev libssl-dev ladspa-sdk swh-plugins cmt tap-plugins
    libpthread-stubs0-dev
)

# --- Install Missing Packages ---
echo "Checking for installed packages..."
MISSING_PKGS=()
for pkg in "${PACKAGES[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
        MISSING_PKGS+=("$pkg")
        echo "Package $pkg will be installed"
    else
        echo "Package $pkg is already installed - skipping"
    fi
done

if [ ${#MISSING_PKGS[@]} -ne 0 ]; then
    echo "Installing missing packages..."
    sudo apt install -y "${MISSING_PKGS[@]}"
else
    echo "All required packages are already installed"
fi

# --- User & Group Setup ---
echo "Setting up user and group permissions..."
sudo groupadd -f root
if ! id "root" &>/dev/null; then
    sudo useradd -r -g root -d /etc/root root
fi
sudo usermod -aG audio,daemon,dialout,gpio,plugdev root

# --- Backup existing installation ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -d "/etc/root" ]; then
    echo "Creating backup of /etc/root..."
    sudo cp -a /etc/root "/etc/root.backup_$TIMESTAMP"
fi
if [ -d "/usr/local/src/root" ]; then
    echo "Creating backup of source directory..."
    sudo cp -a /usr/local/src/root "/usr/local/src/root.backup_$TIMESTAMP"
fi

# --- Compilation ---
echo "Compiling project..."

# Ensure source directory exists
SRC_DIR="/usr/local/src/root"
if [ -d "$SRC_DIR" ]; then
    echo "Found existing source - updating..."
    sudo chown -R root:root "$SRC_DIR"
    cd "$SRC_DIR"
    sudo  git reset --hard
    sudo  git clean -fd
else
    sudo mkdir -p /usr/local/src
    sudo chmod 777 /usr/local/src
    cd /usr/local/src
    sudo  git clone https://github.com/sm0svx/svxlink.git root
fi

cd "$SRC_DIR"
sudo  git config --global --add safe.directory "$SRC_DIR"
sudo  git pull

# --- Build ---
echo "Building..."
sudo chown -R root:root "$SRC_DIR"
mkdir -p "$SRC_DIR/build"
cd "$SRC_DIR/build"

sudo cmake -DUSE_QT=OFF -DCMAKE_INSTALL_PREFIX=/usr \
    -DSYSCONF_INSTALL_DIR=/etc -DLOCAL_STATE_DIR=/var \
    -DWITH_SYSTEMD=ON ../src

# Determine optimal number of build jobs
CORES=$(nproc)
MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)

if [ "$MEM_MB" -lt 700 ]; then
    JOBS=1
elif [ "$CORES" -le 2 ]; then
    JOBS=$CORES
else
    JOBS=$((CORES - 1))
fi

echo "Compiling using $JOBS threads on $CORES cores with ${MEM_MB}MB RAM"
sudo make -j"$JOBS"

# --- Install ---
sudo make doc
sudo make install
sudo ldconfig

# --- Reload Systemd and Restart ---
echo "Reloading system services..."
sudo systemctl daemon-reload
sleep 5
sudo systemctl restart root

# --- Show log output ---
sudo journalctl  --no-pager -n 150

echo "###-FINISH-###"
