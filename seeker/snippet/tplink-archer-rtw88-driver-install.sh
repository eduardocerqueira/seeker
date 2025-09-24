#date: 2025-09-24T17:05:58Z
#url: https://api.github.com/gists/7079f6529c66c85a77575eed78be460a
#owner: https://api.github.com/users/gustavomdsantos

#!/usr/bin/env bash

#
# tplink-archer-rtw88-driver-install.sh - Setup script for TP-Link
# Archer T2U Nano (AC600 model and similar, RTL8811AU chipset) driver for
# Debian-based systems (Linux kernel v6.14 or older).
#
# This script is NOT needed in Linux systems with kernel v6.14 or newer.
# USB Wi-Fi Adapters with RTL88* chipsets are PnP in these systems.
#
# Usage:
#   chmod +x tplink-archer-rtw88-driver-install.sh
#   sudo ./tplink-archer-rtw88-driver-install.sh [--help | --uninstall]
#

set -euo pipefail

# --- Configuration ---
REPO_URL="https://github.com/lwfinger/rtw88.git"
BUILD_DIR="/tmp/rtw88-build-$(date +%s)"
DRIVER_MODULE="rtw_8821au"
CONFLICTING_MODULE="rtl8xxxu"
BLACKLIST_FILE="/etc/modprobe.d/blacklist-${CONFLICTING_MODULE}.conf"
USB_VENDOR_ID="2357"
USB_PRODUCT_ID="011e"
OLD_DKMS_DRIVER="8812au"

# --- Helper Functions ---
log() {
    local level="$1"; shift
    case "$level" in
        info)    echo -e "\033[1;34m[INFO]\033[0m $*" ;;
        success) echo -e "\033[1;32m[SUCCESS]\033[0m $*" ;;
        warn)    echo -e "\033[1;33m[WARNING]\033[0m $*" ;;
        error)   echo -e "\033[1;31m[ERROR]\033[0m $*" ;;
    esac
}

check_root() {
    if [[ "$(id -u)" -ne 0 ]]; then
        log error "This script must be run as root. Use: sudo $0"
        exit 1
    fi
}

show_help() {
    local msg="
TP-Link Archer T2U Nano (AC600, RTL8811AU chipset) driver installer.

This script automates:
  - Installing dependencies
  - Blacklisting conflicting modules
  - Building and installing rtw88_usb driver
  - Verifying device detection

Usage:
  sudo $0                Install the driver
  sudo $0 --uninstall    Uninstall the driver and revert changes
  $0 --help              Show this help message

Notes:
  - Requires Debian-based distributions (e.g., Ubuntu, Linux Mint)
  - Not required on kernel v6.14 or newer (driver is built-in)

Author: Gustavo Moraes <gustavomdsantos@pm.me>
"
    echo -e "$msg"
}

# --- Install Logic ---
install_dependencies() {
    log info "Installing build dependencies and kernel headers..."
    apt-get update -qq
    apt-get install -y git make gcc build-essential libelf-dev linux-headers-$(uname -r)
}

blacklist_conflicting_module() {
    log info "Blacklisting conflicting module: '$CONFLICTING_MODULE'..."
    if [ ! -f "$BLACKLIST_FILE" ]; then
        echo "blacklist $CONFLICTING_MODULE" > "$BLACKLIST_FILE"
        log success "Module blacklisted at $BLACKLIST_FILE. This is permanent."
    else
        log info "Conflicting module already blacklisted."
    fi
}

cleanup_previous_attempts() {
    log info "Unloading potentially active drivers..."
    # Unload all possible conflicting modules. Ignore errors if they aren't loaded.
    modprobe -r "$CONFLICTING_MODULE" 2>/dev/null || true
    modprobe -r 8812au 88x2bu 2>/dev/null || true
    modprobe -r "$DRIVER_MODULE" 2>/dev/null || true

    # Remove old DKMS driver if it exists
    if dkms status | grep -q "$OLD_DKMS_DRIVER"; then
        log warn "Removing previous '$OLD_DKMS_DRIVER' DKMS module..."
        DKMS_TO_REMOVE=$(dkms status | grep "$OLD_DKMS_DRIVER" | tr -d ',' | awk '{print $1"/"$2}')
        dkms remove "$DKMS_TO_REMOVE" --all || true
        rm -rf "/usr/src/$(basename "$DKMS_TO_REMOVE")" || true
    fi
}

install_driver() {
    log info "Cloning lwfinger/rtw88 driver repository..."
    rm -rf "$BUILD_DIR"
    git clone --depth 1 "$REPO_URL" "$BUILD_DIR"

    cd "$BUILD_DIR"
    log info "Building the driver module..."
    make

    log info "Installing the driver module..."
    make install

    # Refresh module dependencies database
    depmod -a
    log success "Driver '$DRIVER_MODULE' has been built and installed."
}

load_and_verify() {
    log info "Attempting to load the new driver module: $DRIVER_MODULE"
    if ! modprobe "$DRIVER_MODULE"; then
        log error "Failed to load '$DRIVER_MODULE'. Check 'dmesg' for clues."
        exit 1
    fi
    log success "Driver module '$DRIVER_MODULE' loaded successfully."

    log info "Triggering udev to associate driver with device..."
    udevadm trigger

    # Give the system a few seconds to create the network interface
    sleep 3

    log info "Verifying adapter and network interface..."
    if lsusb | grep -qi "${USB_VENDOR_ID}:${USB_PRODUCT_ID}"; then
        log info "TP-Link Archer T2U Nano adapter detected."
        if ip link | grep -qE "wlan|wlx"; then
            WLAN_INTERFACE=$(ip link | grep -oE "(wlan[0-9]+|wlx[a-f0-9]+)")
            log success "Network interface '$WLAN_INTERFACE' is UP! Check your Network Manager."
        else
            log error "Adapter detected, but no wireless interface was created."
            log info "Run 'dmesg | tail -n 30' for details."
        fi
    else
        log warn "Adapter not detected via lsusb. Ensure it is plugged in."
    fi
}

# --- Uninstall Logic ---
uninstall_driver() {
    check_root
    log info "Starting uninstallation process..."

    if lsmod | grep -q "^${DRIVER_MODULE}"; then
        log info "Unloading driver module $DRIVER_MODULE..."
        modprobe -r "$DRIVER_MODULE" || true
    fi

    if [ -d "$BUILD_DIR" ]; then
        log info "Removing build directory $BUILD_DIR..."
        rm -rf "$BUILD_DIR"
    fi

    log info "Removing installed driver module files..."
    DRIVER_KO_FILES=$(find /lib/modules/$(uname -r) -type f -name "${DRIVER_MODULE}*.ko*" 2>/dev/null)
    if [ -n "$DRIVER_KO_FILES" ]; then
        for file in $DRIVER_KO_FILES; do
            log info "Removing $file..."
            rm -f "$file"
        done
    else
        log info "No module files found for $DRIVER_MODULE."
    fi

    if [ -f "$BLACKLIST_FILE" ]; then
        log info "Removing blacklist file $BLACKLIST_FILE..."
        rm -f "$BLACKLIST_FILE"
    fi

    log info "Updating module dependencies..."
    depmod -a

    log info "Triggering udev to refresh device state..."
    udevadm trigger || true

    log success "Uninstallation complete. If the adapter still shows up, please reboot to fully unload the driver from memory."
}


# --- Main Execution ---
main() {
    if [[ $# -eq 1 ]]; then
        case "$1" in
            --help)
                show_help
                exit 0
                ;;
            --uninstall)
                uninstall_driver
                exit 0
                ;;
            *)
                log error "Unknown option: $1"
                echo "Use --help for usage."
                exit 1
                ;;
        esac
    elif [[ $# -eq 0 ]]; then
        check_root
        install_dependencies
        blacklist_conflicting_module
        cleanup_previous_attempts
        install_driver
        load_and_verify
        log success "Script finished. A restart is recommended."
    else
        log error "Too many arguments."
        echo "Use --help for usage."
        exit 1
    fi
}

main "$@"

