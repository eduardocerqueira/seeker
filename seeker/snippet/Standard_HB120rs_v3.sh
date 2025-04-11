#date: 2025-04-11T16:46:41Z
#url: https://api.github.com/gists/7e842929d864593c08640433a7407885
#owner: https://api.github.com/users/ConnorBaker

#!/usr/bin/env bash

# General script used to set up HB120rs_v3 instances, don't remember if it conflicts with mellanox.sh

set -euo pipefail

declare -r USER="azureuser"
declare -r USER_HOME="/home/$USER"
declare -r BTRFS_BLOCK_DEVICE="/dev/nvme0n1"
declare -r BTRFS_MOUNT="/mnt/fs"

_log() {
  if (($# != 2)); then
    echo "_log: missing function name and message" >&2
    exit 1
  fi
  echo "[$(date)][${1:?}] ${2:?}"
}

installPrerequisites() {
  log() { _log "installPrerequisites" "$@"; }
  local -ar packages=(
    # Btrfs
    "btrfs-progs"
    "gdisk"
    # bpftune
    # See https://github.com/oracle/bpftune?tab=readme-ov-file#getting-started
    "make"
    "libbpf1"
    "libbpf-dev"
    "libcap-dev"
    "linux-tools-common" # Provides bpftool
    "libnl-route-3-dev"  # TODO: This wasn't one of the dependencies listed in the README
    "libnl-3-dev"
    "clang"
    "llvm"
    "python3-docutils"
    # ZRAM
    "linux-modules-extra-azure"
    "zstd"
    # Generally required
    "git"
    "gpg"
    "inetutils-ping"
  )

  log "Updating apt"
  sudo apt-get update

  log "Installing packages: ${packages[*]}"
  sudo apt-get install -y "${packages[@]}"
}

setupZramSwap() {
  log() { _log "setupZramSwap" "$@"; }
  local -r swapSize="1TB"

  log "Enabling zram module"
  sudo modprobe zram num_devices=1

  log "Creating zram0 device"
  sudo zramctl --find --size "$swapSize" --algorithm zstd

  log "Enabling zram0 device"
  sleep 2
  sudo mkswap /dev/zram0
  sleep 2
  sudo swapon --priority -2 /dev/zram0
}

setupBtrfsMntFsVolume() {
  log() { _log "setupBtrfsMntFsVolume" "$@"; }
  local -ar disks=(
    "/dev/nvme0n1"
    "/dev/nvme1n1"
  )

  log "Creating Btrfs volume"
  for disk in "${disks[@]}"; do
    log "Processing $disk"

    log "Wiping disk"
    sudo sgdisk --zap-all "$disk"

    log "Creating GPT"
    sudo parted --script "$disk" mklabel gpt mkpart primary 0% 100%
  done

  log "Waiting for device nodes to appear"
  sudo udevadm settle

  log "Formatting disks"
  sudo mkfs.btrfs --force --label fs --data raid0 "${disks[@]}"

  log "Mounting disks"
  sudo mkdir -p "$BTRFS_MOUNT"
  sudo mount -t btrfs -o defaults,noatime "$BTRFS_BLOCK_DEVICE" "$BTRFS_MOUNT"
}

createBtrfsMntFsSubvolume() {
  log() { _log "createBtrfsMntFsSubvolume" "$@"; }
  if (($# != 2)); then
    log "!!! missing subvolume name and path !!!" >&2
    exit 1
  fi
  local -r name="$1"
  local -r mountPoint="$2"

  log "Creating subvolume $name"
  sudo btrfs subvolume create "$BTRFS_MOUNT/$name"

  log "Mounting subvolume $name"
  sudo mkdir -p "$mountPoint" "$BTRFS_MOUNT/$name"
  sudo mount -t btrfs -o defaults,noatime,subvol="$name" "$BTRFS_BLOCK_DEVICE" "$mountPoint"
}

setupBtrfsMntFsSubvolumes() {
  log() { _log "setupBtrfsMntFsSubvolumes" "$@"; }
  local -ar subvolumeNames=(
    "nix"
    "tmp"
    "working"
  )

  log "Creating Btrfs subvolumes"
  for name in "${subvolumeNames[@]}"; do
    createBtrfsMntFsSubvolume "$name" "/$name"
  done

  log "Fixing permissions on /tmp"
  sudo chmod -R 1777 "/tmp"

  log "Fixing permissions on /working"
  sudo chown -R "$USER:$USER" "/working"

  log "Setting up an OverlayFS mount for $USER_HOME on /working"
  local -r lowerDir="/home/.$USER"
  local -r upperDir="/working/$USER"
  local -r workDir="/working/.$USER"
  sudo mv "$USER_HOME" "$lowerDir"
  sudo mkdir -p "$upperDir" "$workDir" "$USER_HOME"
  sudo mount -t overlay overlay -o lowerdir="$lowerDir",upperdir="$upperDir",workdir="$workDir" "$USER_HOME"

  log "Setting permissions on $USER_HOME"
  sudo chown -R "$USER:$USER" "$USER_HOME"
}

setupNix() {
  log() { _log "setupNix" "$@"; }
  local -r NIX_CONFIG="/etc/nix/nix.conf"
  local -r NIX_INSTALLER="https://install.determinate.systems/nix"
  local -r NIX_VERSION="2.25.3"
  local -r NIX_PLATFORM="x86_64-linux"
  local -r NIX_PACKAGE_URL="https://releases.nixos.org/nix/nix-$NIX_VERSION/nix-$NIX_VERSION-$NIX_PLATFORM.tar.xz"
  local -ra extraConfig=(
    "accept-flake-config = true"
    "allow-import-from-derivation = false"
    "auto-allocate-uids = true"
    "builders-use-substitutes = true"
    "connect-timeout = 10"
    "experimental-features = auto-allocate-uids cgroups flakes mounted-ssh-store nix-command"
    "fsync-metadata = false"
    "http-connections = 128"
    "log-lines = 100"
    "max-substitution-jobs = 64"
    "narinfo-cache-negative-ttl = 0"
    "sandbox-fallback = false"
    "substituters = https://cache.nixos.org"
    "trusted-public-keys = cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
    "trusted-users = root runner @wheel"
    "use-cgroups = true"
    "use-xdg-base-directories = true"
    "warn-dirty = false"
  )

  log "Installing Nix $NIX_VERSION for $NIX_PLATFORM"
  curl --proto '=https' --tlsv1.2 -sSf -L "$NIX_INSTALLER" |
    sh -s -- install --no-confirm --nix-package-url "$NIX_PACKAGE_URL"

  log "Overriding defaults in Nix configuration"
  sudo sed \
    -e '/auto-optimise-store =/d' \
    -e '/experimental-features =/d' \
    -e '/upgrade-nix-store-path-url =/d' \
    -i "$NIX_CONFIG"

  log "Adding extra Nix configuration"
  for line in "${extraConfig[@]}"; do
    echo "$line" | sudo tee -a "$NIX_CONFIG"
  done

  log "Reloading Nix configuration"
  sudo systemctl restart nix-daemon
}

setupKernelVmParameters() {
  log() { _log "setupKernelVmParameters" "$@"; }

  # Taken from: https://github.com/ConnorBaker/nixos-configs/blob/e6d3e54ed9d257bd148a5bfb57dc476570b5d9f0/modules/zram.nix
  local -ra vmParameters=(
    # https://wiki.archlinux.org/title/Zram#Optimizing_swap_on_zram
    "vm.watermark_boost_factor=0"
    "vm.watermark_scale_factor=125"
    "vm.page-cluster=0"

    # https://github.com/pop-os/default-settings/blob/master_noble/etc/sysctl.d/10-pop-default-settings.conf
    "vm.swappiness=190" # Strong preference for ZRAM
    "vm.max_map_count=2147483642"

    # Higher values since these machines are used mostly as remote builders
    "vm.dirty_ratio=80"
    "vm.dirty_background_ratio=50"
  )

  log "Setting up kernel VM parameters"
  for param in "${vmParameters[@]}"; do
    sudo sysctl -w "$param"
  done
}

setupKernelNetParameters() {
  log() { _log "setupKernelNetParameters" "$@"; }

  # Taken from: https://github.com/ConnorBaker/nixos-configs/blob/e6d3e54ed9d257bd148a5bfb57dc476570b5d9f0/modules/networking.nix
  local -ri KB=1024
  local -ri MB=$((KB * KB))

  # Memory settings
  local -ri memMin=$((8 * KB))
  local -ri rmemDefault=$((128 * KB))
  local -ri wmemDefault=$((16 * KB))
  local -ri memMax=$((16 * MB))

  local -ra netParameters=(
    # Enable BPF JIT for better performance
    "net.core.bpf_jit_enable=1"
    "net.core.bpf_jit_harden=0"

    # Change the default queueing discipline to cake and the congestion control algorithm to BBR
    "net.core.default_qdisc=cake"
    "net.ipv4.tcp_congestion_control=bbr"

    # Largely taken from https://wiki.archlinux.org/title/sysctl and
    # https://github.com/redhat-performance/tuned/blob/master/profiles/network-throughput/tuned.conf#L10
    "net.core.somaxconn=$((8 * KB))"
    "net.core.netdev_max_backlog=$((16 * KB))"
    "net.core.optmem_max=$((64 * KB))"

    # RMEM
    "net.core.rmem_default=$rmemDefault"
    "net.core.rmem_max=$memMax"
    "net.ipv4.tcp_rmem=$memMin $rmemDefault $memMax"
    "net.ipv4.udp_rmem_min=$memMin"

    # WMEM
    "net.core.wmem_default=$wmemDefault"
    "net.core.wmem_max=$memMax"
    "net.ipv4.tcp_wmem=$memMin $wmemDefault $memMax"
    "net.ipv4.udp_wmem_min=$memMin"

    # General TCP
    "net.ipv4.tcp_fastopen=3"
    "net.ipv4.tcp_fin_timeout=10"
    "net.ipv4.tcp_keepalive_intvl=10"
    "net.ipv4.tcp_keepalive_probes=6"
    "net.ipv4.tcp_keepalive_time=60"
    "net.ipv4.tcp_max_syn_backlog=$((8 * KB))"
    "net.ipv4.tcp_max_tw_buckets=2000000"
    "net.ipv4.tcp_mtu_probing=1"
    "net.ipv4.tcp_slow_start_after_idle=0"
    "net.ipv4.tcp_tw_reuse=1"
  )

  log "Setting up kernel network parameters"
  for param in "${netParameters[@]}"; do
    sudo sysctl -w "$param"
  done
}

setupBpftune() {
  log() { _log "setupBpftune" "$@"; }
  local -r BASE_URL="https://github.com/oracle/bpftune/archive"
  local -r REV="0e6bca2e5880fcbaac6478c4042f5f9314e61463"
  local -r TARBALL_NAME="bpftune-$REV.tar.gz"
  local -r BPFTUNE_DIR="$USER_HOME/bpftune"

  log "Creating directory for bpftune"
  mkdir -p "$BPFTUNE_DIR"

  log "Entering directory for bpftune"
  pushd "$BPFTUNE_DIR"

  log "Downloading bpftune tarball"
  curl --location "$BASE_URL/$REV.tar.gz" --output "$TARBALL_NAME"

  log "Extracting bpftune tarball"
  tar xzf "$TARBALL_NAME" --strip-components=1

  log "Removing downloaded archive"
  rm -f "$TARBALL_NAME"

  log "Building bpftune"
  make -j

  log "Installing bpftune"
  sudo make install

  log "Staring bpftune"
  sudo systemctl enable bpftune
  sudo systemctl start bpftune

  log "Exiting directory for bpftune"
  popd
}


main() {
  # Software
  installPrerequisites

  # Memory
  setupZramSwap
  setupKernelVmParameters # Values chosen for ZRAM

  # Disks
  setupBtrfsMntFsVolume
  setupBtrfsMntFsSubvolumes

  # Nix
  setupNix

  # Network
  setupKernelNetParameters
  # setupBpftune
}

main