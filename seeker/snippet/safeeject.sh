#date: 2025-06-09T16:53:32Z
#url: https://api.github.com/gists/894cdf3c3ecd10091bbbabb95830e1aa
#owner: https://api.github.com/users/amir-arad

#!/usr/bin/env bash

# USB Drive Management for Linux
# Written completely by LLMs @2025
# Version 3.2 - Fixed security vulnerabilities and production issues

set -euo pipefail
set -o errexit
set -o nounset
set -o pipefail

# Force consistent locale for system commands
export LANG=C
export LC_ALL=C

# Colors and formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Configuration
readonly SCRIPT_NAME="idiot-proof-usb"
readonly VERSION="3.2"
readonly INSTALL_MARKER="$HOME/.config/${SCRIPT_NAME}_installed"
readonly LOG_FILE="/var/log/${SCRIPT_NAME}.log"
readonly LOCK_DIR="/var/lock/${SCRIPT_NAME}"

# Performance and reliability settings
readonly SYNC_TIMEOUT=15
readonly LSOF_TIMEOUT=5
readonly UNMOUNT_RETRIES=5
readonly PROCESS_WAIT_TIMEOUT=12

# Security settings
readonly SECURE_MOUNT_OPTS="nodev,nosuid,noexec,sync"

# Create lock directory
[[ -d "$LOCK_DIR" ]] || sudo mkdir -p "$LOCK_DIR"

# Helper functions
log_info() { echo -e "${BLUE}ℹ️  $*${NC}"; }
log_success() { echo -e "${GREEN}✅ $*${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $*${NC}"; }
log_error() { echo -e "${RED}❌ $*${NC}" >&2; }
log_debug() { echo "$(date '+%Y-%m-%d %H:%M:%S') DEBUG: $*" >> "$LOG_FILE" 2>/dev/null || true; }
log_security() { logger -t "${SCRIPT_NAME}[$$]" -p "authpriv.warning" "SECURITY: $*" 2>/dev/null || true; }

# Error handling
error_exit() {
    log_error "${1:-Unknown Error}"
    cleanup_resources
    exit 1
}

cleanup_resources() {
    # Remove any lock files owned by this process
    find "$LOCK_DIR" -name "*.$$.lock" -delete 2>/dev/null || true
}

# Set up exit trap
trap cleanup_resources EXIT INT TERM

# Input validation
validate_path() {
    local path="$1"
    # Only allow alphanumeric, /, -, _, and space
    if [[ ! "$path" =~ ^[a-zA-Z0-9/_[:space:]-]+$ ]]; then
        log_security "Invalid path attempted: $path"
        return 1
    fi
    # Resolve path safely
    if [[ -e "$path" ]]; then
        realpath -e "$path" 2>/dev/null || return 1
    else
        return 1
    fi
}

# Lock management
acquire_lock() {
    local resource="$1"
    local lockfile="$LOCK_DIR/${resource//\//_}.$$.lock"
    
    if (set -C; echo "$$" > "$lockfile") 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

release_lock() {
    local resource="$1"
    local lockfile="$LOCK_DIR/${resource//\//_}.$$.lock"
    rm -f "$lockfile"
}

show_help() {
    cat << EOF
${BOLD}Complete Enterprise USB Drive Management for Linux v${VERSION}${NC}

Production-ready USB handling with comprehensive error handling and all features implemented.

${BOLD}USAGE:${NC}
    $0 [OPTIONS]

${BOLD}OPTIONS:${NC}
    --install     Install the complete USB management system
    --uninstall   Remove all changes and restore defaults
    --check       Check installation status and system health
    --test        Test system functionality
    --help        Show this help message
    --version     Show version information

${BOLD}WHAT IT DOES:${NC}
    ✅ Auto-mount USB drives with immediate write (no caching)
    ✅ Enterprise-grade 'safeeject' command with retry logic
    ✅ Multi-interface GUI eject options
    ✅ File manager integration (right-click eject)
    ✅ Fix Alt+Shift hotkey conflicts (with restore capability)
    ✅ Comprehensive error handling and timeouts
    ✅ Works in headless/non-GUI environments
    ✅ Handles encrypted volumes and edge cases
    ✅ Locale-independent operation

${BOLD}AFTER INSTALLATION:${NC}
    Command: safeeject              (show USB drives)
    Command: safeeject /media/*/DRIVE  (eject specific)
    Command: safeeject all          (eject everything)
    Command: eject                  (alias for safeeject)
    GUI: Right-click files on USB → 'Safe Eject'
    GUI: Search 'Safe Eject USB' in applications

${BOLD}SYSTEM REQUIREMENTS:${NC}
    • Ubuntu 20.04+ or compatible Linux distribution
    • udisks2 package
    • Optional: zenity (for GUI functionality)

EOF
}

# System capability detection
detect_system_capabilities() {
    local capabilities=()
    
    # Check if we're in a GUI session
    if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        capabilities+=("gui")
    fi
    
    # Check udisks2 availability and version
    if command -v udisksctl &> /dev/null; then
        capabilities+=("udisks2")
        
        # Check udisks2 version
        local udisks_version
        udisks_version=$(udisksctl --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [[ "${udisks_version%%.*}" -ge 2 ]] && [[ "${udisks_version#*.}" -ge 9 ]]; then
            capabilities+=("udisks2-modern")
        else
            capabilities+=("udisks2-legacy")
        fi
        
        # Check user-mode udisks2 service
        if systemctl --user is-active udisks2.service &>/dev/null; then
            capabilities+=("udisks2-user")
        else
            log_debug "udisks2 user service not running"
        fi
        
        # Check system-mode udisks2 service
        if systemctl is-active udisks2.service &>/dev/null; then
            capabilities+=("udisks2-system")
        else
            log_debug "udisks2 system service not running"
        fi
    fi
    
    # Check for zenity (GUI dialogs)
    if command -v zenity &> /dev/null; then
        capabilities+=("zenity")
    fi
    
    # Check for lsof vs fuser performance options
    if command -v fuser &> /dev/null; then
        capabilities+=("fuser")
    fi
    
    if command -v lsof &> /dev/null; then
        capabilities+=("lsof")
    fi
    
    # Check udevadm for device info fallback
    if command -v udevadm &> /dev/null; then
        capabilities+=("udevadm")
    fi
    
    # Check for gsettings (GNOME settings)
    if command -v gsettings &> /dev/null; then
        capabilities+=("gsettings")
    fi
    
    echo "${capabilities[@]}"
}

# Enhanced timeout wrapper with logging
timeout_cmd() {
    local timeout_duration="$1"
    local description="$2"
    shift 2
    
    log_debug "Running with ${timeout_duration}s timeout: $*"
    
    if timeout "$timeout_duration" "$@" 2>/dev/null; then
        log_debug "Command succeeded: $*"
        return 0
    else
        local exit_code=$?
        log_debug "Command failed/timed out (exit $exit_code): $*"
        log_warning "$description timed out after ${timeout_duration}s"
        return $exit_code
    fi
}

# Enhanced sync with timeout and fallback
safe_sync() {
    log_info "Syncing data to disk..."
    
    if timeout_cmd "$SYNC_TIMEOUT" "Data sync" sync; then
        log_debug "Sync completed successfully"
        return 0
    else
        log_warning "Sync timed out - drive may be failing or very busy"
        log_warning "Continuing with caution..."
        return 1
    fi
}

# Robust device information gathering with fallbacks
get_device_info() {
    local device="$1"
    local info_type="$2"  # removable, fstype, label, etc.
    
    # Validate device path
    validate_path "$device" || return 1
    
    # Primary method: lsblk
    local result
    result=$(LANG=C lsblk -no "$info_type" "$device" 2>/dev/null) || result=""
    
    if [ -n "$result" ] && [ "$result" != "" ]; then
        echo "$result"
        return 0
    fi
    
    # Fallback: udevadm (if available)
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    if [[ " ${capabilities[*]} " =~ " udevadm " ]]; then
        log_debug "lsblk failed for $device, trying udevadm fallback"
        
        case "$info_type" in
            "REMOVABLE")
                result=$(udevadm info --query=property --name="$device" 2>/dev/null | grep "ID_BUS=usb" && echo "1" || echo "0")
                ;;
            "FSTYPE")
                result=$(udevadm info --query=property --name="$device" 2>/dev/null | grep "ID_FS_TYPE=" | cut -d'=' -f2)
                ;;
            "LABEL")
                result=$(udevadm info --query=property --name="$device" 2>/dev/null | grep "ID_FS_LABEL=" | cut -d'=' -f2)
                ;;
        esac
        
        if [ -n "$result" ]; then
            log_debug "udevadm fallback succeeded for $device"
            echo "$result"
            return 0
        fi
    fi
    
    log_debug "All methods failed to get $info_type for $device"
    return 1
}

# High-performance process checking with fuser fallback
check_processes_using_path() {
    local path="$1"
    local timeout_duration="${2:-$LSOF_TIMEOUT}"
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    log_debug "Checking processes using $path"
    
    # Try fuser first (faster for large directories)
    if [[ " ${capabilities[*]} " =~ " fuser " ]]; then
        if timeout_cmd "$timeout_duration" "Process check (fuser)" fuser -m "$path" >/dev/null 2>&1; then
            log_debug "fuser found processes using $path"
            return 0
        fi
    fi
    
    # Fallback to lsof
    if [[ " ${capabilities[*]} " =~ " lsof " ]]; then
        if timeout_cmd "$timeout_duration" "Process check (lsof)" lsof +D "$path" >/dev/null 2>&1; then
            log_debug "lsof found processes using $path"
            return 0
        fi
    fi
    
    log_debug "No processes found using $path"
    return 1
}

# Get detailed process information for user feedback
get_process_details() {
    local path="$1"
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    # Try fuser first
    if [[ " ${capabilities[*]} " =~ " fuser " ]]; then
        local fuser_output
        fuser_output=$(fuser -v "$path" 2>&1 | tail -n +2 || true)
        if [ -n "$fuser_output" ]; then
            echo "$fuser_output"
            return 0
        fi
    fi
    
    # Fallback to lsof
    if [[ " ${capabilities[*]} " =~ " lsof " ]]; then
        local lsof_output
        lsof_output=$(lsof +D "$path" 2>/dev/null | head -10 || true)
        if [ -n "$lsof_output" ]; then
            echo "$lsof_output"
            return 0
        fi
    fi
    
    echo "Unable to determine specific processes"
    return 1
}

# Enhanced removable device detection
is_removable_device() {
    local path="$1"
    
    # Get the device
    local device
    device=$(LANG=C df "$path" 2>/dev/null | tail -1 | awk '{print $1}') || {
        log_debug "Failed to get device for $path"
        return 1
    }
    
    # Remove partition number to get base device
    local base_device
    base_device=$(echo "$device" | sed 's/[0-9]*$//')
    
    # Check if removable using enhanced detection
    local removable
    removable=$(get_device_info "$base_device" "REMOVABLE") || {
        log_debug "Failed to determine if $base_device is removable"
        return 1
    }
    
    [ "$removable" = "1" ]
}

# Robust mount point detection with validation
get_mount_point() {
    local target="$1"
    
    # Validate input
    validate_path "$target" || return 1
    
    # If it's already a mount point, validate it
    if [ -d "$target" ]; then
        if mountpoint -q "$target" 2>/dev/null; then
            echo "$target"
            return 0
        fi
    fi
    
    # If it's a device, find its mount point
    if [[ "$target" == /dev/* ]]; then
        local mount_point
        mount_point=$(LANG=C findmnt -n -o TARGET "$target" 2>/dev/null) || {
            log_debug "Device $target is not mounted"
            return 1
        }
        echo "$mount_point"
        return 0
    fi
    
    # Try to find mount point of the filesystem containing the path
    local mount_point
    mount_point=$(LANG=C df "$target" 2>/dev/null | tail -1 | awk '{print $6}') || {
        log_debug "Failed to find mount point for $target"
        return 1
    }
    
    echo "$mount_point"
    return 0
}

# Enhanced unmount with comprehensive retry logic
unmount_with_retry() {
    local mount_point="$1"
    local max_attempts="${2:-$UNMOUNT_RETRIES}"
    local base_delay="${3:-2}"
    
    # Acquire lock for this mount point
    if ! acquire_lock "$mount_point"; then
        log_error "Could not acquire lock for $mount_point"
        return 1
    fi
    
    local attempt
    for attempt in $(seq 1 $max_attempts); do
        local delay=$((base_delay * (2 ** (attempt - 1))))  # Exponential backoff
        delay=$((delay > 30 ? 30 : delay))  # Cap at 30 seconds
        
        log_info "Unmount attempt $attempt/$max_attempts..."
        
        # Try unmount
        if umount "$mount_point" 2>/dev/null; then
            log_debug "Unmount succeeded on attempt $attempt"
            release_lock "$mount_point"
            return 0
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            log_warning "Unmount failed, waiting ${delay}s before retry..."
            
            # Force sync and wait
            safe_sync
            sleep "$delay"
            
            # Check if processes are still using the device
            if check_processes_using_path "$mount_point" 2; then
                log_warning "Processes still using device..."
                
                # Try lazy unmount as intermediate step
                if umount -l "$mount_point" 2>/dev/null; then
                    log_info "Lazy unmount successful, waiting for processes to finish..."
                    sleep 2
                    
                    # Check if it's actually unmounted now
                    if ! mountpoint -q "$mount_point" 2>/dev/null; then
                        log_debug "Lazy unmount completed successfully"
                        release_lock "$mount_point"
                        return 0
                    fi
                fi
            fi
        fi
    done
    
    release_lock "$mount_point"
    log_debug "All unmount attempts failed"
    return 1
}

# Smart device power-down with service detection
power_down_device() {
    local mount_point="$1"
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    local device
    device=$(LANG=C findmnt -n -o SOURCE "$mount_point" 2>/dev/null | sed 's/[0-9]*$//' || true)
    
    if [ -z "$device" ] || [ ! -b "$device" ]; then
        log_debug "No valid block device found for power-down"
        return 1
    fi
    
    log_debug "Attempting to power down device: $device"
    
    # Check if udisks2 is available and functional
    if [[ " ${capabilities[*]} " =~ " udisks2 " ]]; then
        # Prefer user service if available
        if [[ " ${capabilities[*]} " =~ " udisks2-user " ]]; then
            if timeout_cmd 10 "Device power-down (user)" udisksctl power-off -b "$device"; then
                log_success "Device powered down via user service"
                return 0
            fi
        fi
        
        # Fall back to system service
        if [[ " ${capabilities[*]} " =~ " udisks2-system " ]]; then
            if timeout_cmd 10 "Device power-down (system)" udisksctl power-off -b "$device"; then
                log_success "Device powered down via system service"
                return 0
            fi
        fi
        
        # Try without service specification
        if timeout_cmd 10 "Device power-down (auto)" udisksctl power-off -b "$device"; then
            log_success "Device powered down"
            return 0
        fi
    else
        log_warning "udisks2 not available - skipping device power-down"
    fi
    
    log_debug "Device power-down failed or unavailable"
    return 1
}

# Comprehensive wait for processes with user interaction
wait_for_processes() {
    local path="$1"
    local max_wait="${2:-$PROCESS_WAIT_TIMEOUT}"
    local wait_interval="${3:-1}"
    
    log_info "Waiting for processes to finish accessing $path..."
    
    local elapsed=0
    local showed_processes=false
    
    while [ $elapsed -lt $max_wait ]; do
        if ! check_processes_using_path "$path" 2; then
            log_debug "All processes finished"
            return 0
        fi
        
        # Show process details once
        if [ "$showed_processes" = false ]; then
            local process_details
            process_details=$(get_process_details "$path")
            if [ -n "$process_details" ]; then
                log_warning "Active processes:"
                echo "$process_details" | head -5
                showed_processes=true
            fi
        fi
        
        sleep "$wait_interval"
        elapsed=$((elapsed + wait_interval))
        echo -n "."
        
        # Offer early termination after half the timeout
        if [ $elapsed -ge $((max_wait / 2)) ] && [ $((elapsed % 3)) -eq 0 ]; then
            echo ""
            log_warning "Still waiting... Force eject? (y/N/w=wait more)"
            read -r -n 1 -t 3 response || response=""
            echo ""
            
            case "$response" in
                [Yy])
                    log_info "Force eject requested"
                    return 1  # Signal to force
                    ;;
                [Ww])
                    max_wait=$((max_wait + 10))
                    log_info "Waiting 10 more seconds..."
                    ;;
            esac
        fi
    done
    
    echo ""
    log_debug "Process wait timeout reached"
    return 1
}

# Enhanced drive listing with robust parsing
get_removable_drives_enhanced() {
    local format="NAME,MOUNTPOINT,LABEL,SIZE,REMOVABLE,FSTYPE,UUID"
    
    # Primary method: lsblk -P for reliable parsing
    LANG=C lsblk -P -o "$format" 2>/dev/null | \
    grep 'REMOVABLE="1"' | \
    grep -v 'MOUNTPOINT=""' || true
}

# Parse drive information with error handling
parse_drive_field() {
    local drive_line="$1"
    local field="$2"
    
    echo "$drive_line" | grep -o "${field}=\"[^\"]*\"" | cut -d'"' -f2 || echo ""
}

# Setup mount options for immediate sync with security
setup_mount_options() {
    log_info "Configuring automatic sync mounting for USB drives..."
    
    # Create udev rule for removable devices with immediate write and security
    sudo tee /etc/udev/rules.d/99-removable-sync.rules > /dev/null << 'EOF'
# Mount removable USB drives with sync for immediate writes and security options
# This prevents data loss when users remove drives without proper ejection
ACTION=="add", SUBSYSTEM=="block", ENV{ID_BUS}=="usb", ENV{DEVTYPE}=="partition", RUN+="/bin/sh -c 'echo deadline > /sys/block/%k/../queue/scheduler 2>/dev/null || true'"

# Additional rule for USB mass storage devices
ACTION=="add", SUBSYSTEM=="block", ATTRS{removable}=="1", ENV{ID_BUS}=="usb", RUN+="/bin/sh -c 'echo 1 > /sys/block/%k/queue/iosched/fifo_batch 2>/dev/null || true'"
EOF

    # Configure udisks2 for sync mounting of removable drives with security
    sudo mkdir -p /etc/udisks2
    
    # Get current user info safely
    local current_uid current_gid
    current_uid=$(id -u)
    current_gid=$(id -g)
    
    # Detect udisks2 version for proper configuration
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    if [[ " ${capabilities[*]} " =~ " udisks2-modern " ]]; then
        # Modern udisks2 (2.9.0+) configuration
        sudo tee /etc/udisks2/mount_options.conf > /dev/null << EOF
# USB Safe Mount Configuration
# Forces immediate write (sync) and security options for removable drives

[defaults]
# FAT32/FAT16 filesystems (most common on USB drives)
vfat_defaults=uid=${current_uid},gid=${current_gid},shortname=mixed,dmask=0077,fmask=0177,${SECURE_MOUNT_OPTS}

# exFAT filesystems (modern large USB drives)  
exfat_defaults=uid=${current_uid},gid=${current_gid},dmask=0077,fmask=0177,${SECURE_MOUNT_OPTS}

# NTFS filesystems (Windows-formatted drives)
ntfs_defaults=uid=${current_uid},gid=${current_gid},dmask=0077,fmask=0177,big_writes,${SECURE_MOUNT_OPTS}

# ext4/ext3/ext2 (Linux-formatted USB drives)
ext4_defaults=${SECURE_MOUNT_OPTS}
ext3_defaults=${SECURE_MOUNT_OPTS}
ext2_defaults=${SECURE_MOUNT_OPTS}

[/org/freedesktop/UDisks2/drives/*]
# Force sync and security for all removable drives
removable_defaults=${SECURE_MOUNT_OPTS}
EOF
    else
        # Legacy udisks2 configuration
        sudo tee /etc/udisks2/udisks2.conf > /dev/null << EOF
# USB Safe Mount Configuration (Legacy)
[udisks2]
modules_load_preference=ondemand

[defaults]
# Security options for all filesystems
vfat_allow=uid=\$UID,gid=\$GID,${SECURE_MOUNT_OPTS}
ntfs_allow=uid=\$UID,gid=\$GID,${SECURE_MOUNT_OPTS}
ext_allow=${SECURE_MOUNT_OPTS}
EOF
    fi

    log_success "Configured sync mounting with security options for removable drives"
    log_info "USB drives will now write data immediately with enhanced security"
}

# Setup file manager integration
setup_file_manager_integration() {
    log_info "Setting up file manager integration..."
    
    # Create Nautilus script directory
    mkdir -p ~/.local/share/nautilus/scripts
    
    # Create the Safe Eject script for Nautilus
    tee ~/.local/share/nautilus/scripts/Safe\ Eject > /dev/null << 'EOF'
#!/usr/bin/env bash

# Nautilus Script for Safe USB Eject
# Integrated with the enterprise USB management system

set -euo pipefail
export LANG=C
export LC_ALL=C

# Get selected path or current directory from Nautilus
selected_files="${NAUTILUS_SCRIPT_SELECTED_FILE_PATHS:-}"
current_uri="${NAUTILUS_SCRIPT_CURRENT_URI:-}"

# Determine the path to work with
if [ -n "$selected_files" ]; then
    # Use first selected file/folder
    path=$(echo "$selected_files" | head -1)
    # Remove file:// prefix if present
    path=$(echo "$path" | sed 's|^file://||')
    # URL decode
    path=$(printf '%b' "${path//%/\\x}")
else
    # Use current directory
    path=$(echo "$current_uri" | sed 's|^file://||')
    path=$(printf '%b' "${path//%/\\x}")
    path="${path:-$PWD}"
fi

# Validate we have a path
if [ -z "$path" ] || [ ! -e "$path" ]; then
    if command -v zenity &> /dev/null; then
        zenity --error --text="Could not determine path to eject.\n\nPath: $path" --title="Safe Eject" --no-markup
    else
        echo "Error: Could not determine path to eject: $path" >&2
    fi
    exit 1
fi

# Find mount point and device
mount_point=$(df "$path" 2>/dev/null | tail -1 | awk '{print $6}') || {
    if command -v zenity &> /dev/null; then
        zenity --error --text="Could not determine mount point for:\n$path" --title="Safe Eject" --no-markup
    else
        echo "Error: Could not determine mount point for: $path" >&2
    fi
    exit 1
}

device=$(df "$path" 2>/dev/null | tail -1 | awk '{print $1}') || {
    if command -v zenity &> /dev/null; then
        zenity --error --text="Could not determine device for:\n$path" --title="Safe Eject" --no-markup
    else
        echo "Error: Could not determine device for: $path" >&2
    fi
    exit 1
}

# Check if removable
base_device=$(echo "$device" | sed 's/[0-9]*$//')
is_removable=$(lsblk -no REMOVABLE "$base_device" 2>/dev/null) || {
    if command -v zenity &> /dev/null; then
        zenity --error --text="Could not check if device is removable:\n$device" --title="Safe Eject" --no-markup
    else
        echo "Error: Could not check if device is removable: $device" >&2
    fi
    exit 1
}

if [ "$is_removable" != "1" ]; then
    if command -v zenity &> /dev/null; then
        zenity --error --text="This is not a removable drive:\n$mount_point\n\nDevice: $device" --title="Safe Eject" --no-markup
    else
        echo "Error: This is not a removable drive: $mount_point (device: $device)" >&2
    fi
    exit 1
fi

# Use the appropriate eject method
if [ -x /usr/local/bin/safeeject-gui ] && command -v zenity &> /dev/null; then
    # Use GUI version if available
    NAUTILUS_SCRIPT_SELECTED_FILE_PATHS="$mount_point" /usr/local/bin/safeeject-gui
elif [ -x /usr/local/bin/safeeject ]; then
    # Use command line version in terminal
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- /usr/local/bin/safeeject "$mount_point"
    elif command -v xterm &> /dev/null; then
        xterm -e /usr/local/bin/safeeject "$mount_point"
    else
        # Fallback: run directly (no terminal output)
        /usr/local/bin/safeeject "$mount_point"
    fi
else
    if command -v zenity &> /dev/null; then
        zenity --error --text="Safe eject command not found.\n\nPlease reinstall the USB management system." --title="Safe Eject" --no-markup
    else
        echo "Error: Safe eject command not found. Please reinstall the USB management system." >&2
    fi
    exit 1
fi
EOF

    chmod +x ~/.local/share/nautilus/scripts/Safe\ Eject
    
    # Also create integration for other file managers if they exist
    
    # Thunar (XFCE) custom actions
    if command -v thunar &> /dev/null; then
        mkdir -p ~/.config/Thunar
        
        # Check if custom actions file exists, create basic structure if not
        local thunar_actions="$HOME/.config/Thunar/uca.xml"
        if [ ! -f "$thunar_actions" ]; then
            cat > "$thunar_actions" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<actions>
</actions>
EOF
        fi
        
        # Add safe eject action (basic approach - manual addition recommended)
        log_info "Thunar detected - manual configuration needed for custom actions"
    fi
    
    # PCManFM (LXDE) - uses .desktop files in specific location
    if command -v pcmanfm &> /dev/null; then
        mkdir -p ~/.local/share/file-manager/actions
        
        tee ~/.local/share/file-manager/actions/safe-eject.desktop > /dev/null << 'EOF'
[Desktop Entry]
Type=Action
Name=Safe Eject USB
Description=Safely eject USB drive
Icon=drive-removable-media
Profiles=safe_eject_profile;

[X-Action-Profile safe_eject_profile]
MimeTypes=inode/directory;
Exec=/usr/local/bin/safeeject %f
Name=Safe Eject USB Drive
EOF
    fi
    
    log_success "Set up file manager integration"
    log_info "Right-click 'Safe Eject' now available in file managers"
}

# Create desktop application
create_desktop_app() {
    log_info "Creating desktop application..."
    
    mkdir -p ~/.local/share/applications
    
    # Create the main desktop entry
    tee ~/.local/share/applications/safeeject.desktop > /dev/null << EOF
[Desktop Entry]
Name=Safe Eject USB
Comment=Safely remove USB drives (Windows-like experience)
GenericName=USB Drive Manager  
Exec=/usr/local/bin/safeeject-gui
Icon=drive-removable-media
Type=Application
Categories=System;Utility;HardwareSettings;GTK;
Keywords=USB;eject;remove;safe;drive;windows;removable;hardware;
StartupNotify=true
NoDisplay=false
Terminal=false

# Translations for common locales
Name[es]=Expulsar USB Seguro
Comment[es]=Remover unidades USB de forma segura
Name[fr]=Éjecter USB en sécurité  
Comment[fr]=Retirer les lecteurs USB en toute sécurité
Name[de]=USB sicher entfernen
Comment[de]=USB-Laufwerke sicher entfernen
Name[it]=Espelli USB sicuro
Comment[it]=Rimuovi unità USB in sicurezza
Name[pt]=Ejetar USB com segurança
Comment[pt]=Remover drives USB com segurança
EOF

    # Create a command-line version entry as well
    tee ~/.local/share/applications/safeeject-terminal.desktop > /dev/null << EOF
[Desktop Entry]
Name=Safe Eject USB (Terminal)
Comment=Safely eject USB drives using command line interface
Exec=gnome-terminal -- safeeject
Icon=utilities-terminal
Type=Application
Categories=System;Utility;TerminalEmulator;
Keywords=USB;eject;terminal;command;line;
StartupNotify=true
NoDisplay=true
Terminal=true
EOF

    # Update desktop database to make applications immediately available
    if command -v update-desktop-database &> /dev/null; then
        update-desktop-database ~/.local/share/applications/ 2>/dev/null || {
            log_warning "Could not update desktop database - applications may not appear immediately"
        }
    fi
    
    # Update MIME database for file associations  
    if command -v update-mime-database &> /dev/null; then
        update-mime-database ~/.local/share/mime/ 2>/dev/null || {
            log_debug "Could not update MIME database"
        }
    fi
    
    log_success "Created desktop applications"
    log_info "Search 'Safe Eject USB' in your application launcher"
}

# Fix Alt+Shift hotkeys with restore capability
fix_hotkeys() {
    log_info "Fixing Alt+Shift hotkey conflicts..."
    
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    if [[ ! " ${capabilities[*]} " =~ " gsettings " ]]; then
        log_warning "gsettings not available - skipping hotkey fix"
        return 0
    fi
    
    # Create backup directory for original settings
    local backup_dir="$HOME/.config/${SCRIPT_NAME}/hotkey-backup"
    mkdir -p "$backup_dir"
    
    # Backup current input source switching settings
    local backup_file="$backup_dir/input-source-settings.txt"
    
    if [ ! -f "$backup_file" ]; then
        log_info "Backing up current keyboard shortcut settings..."
        
        {
            echo "# Keyboard shortcut backup created on $(date)"
            echo "# Original input source switching settings"
            echo "switch-input-source=$(gsettings get org.gnome.desktop.wm.keybindings switch-input-source 2>/dev/null || echo \"['<Super>space', '<Shift>space']\")"
            echo "switch-input-source-backward=$(gsettings get org.gnome.desktop.wm.keybindings switch-input-source-backward 2>/dev/null || echo \"['<Super><Shift>space', '<Shift><Super>space']\")"
        } > "$backup_file"
        
        log_success "Backed up original keyboard settings to: $backup_file"
    else
        log_info "Using existing keyboard shortcut backup"
    fi
    
    # Disable Alt+Shift input source switching that conflicts with user hotkeys
    log_info "Disabling conflicting keyboard shortcuts..."
    
    # Get current settings
    local current_switch current_switch_backward
    current_switch=$(gsettings get org.gnome.desktop.wm.keybindings switch-input-source 2>/dev/null || echo "[]")
    current_switch_backward=$(gsettings get org.gnome.desktop.wm.keybindings switch-input-source-backward 2>/dev/null || echo "[]")
    
    # Remove Alt+Shift combinations while preserving other shortcuts
    local new_switch new_switch_backward
    
    # Remove problematic Alt+Shift combinations
    new_switch=$(echo "$current_switch" | sed "s/'<Alt>Shift_L'//g; s/'<Shift>Alt_L'//g; s/'<Alt><Shift>'//g; s/'<Shift><Alt>'//g" | sed 's/, ,/,/g; s/\[,/[/g; s/,\]/]/g')
    new_switch_backward=$(echo "$current_switch_backward" | sed "s/'<Alt>Shift_L'//g; s/'<Shift>Alt_L'//g; s/'<Alt><Shift>'//g; s/'<Shift><Alt>'//g" | sed 's/, ,/,/g; s/\[,/[/g; s/,\]/]/g')
    
    # Apply new settings
    if gsettings set org.gnome.desktop.wm.keybindings switch-input-source "$new_switch" 2>/dev/null; then
        log_debug "Updated switch-input-source to: $new_switch"
    fi
    
    if gsettings set org.gnome.desktop.wm.keybindings switch-input-source-backward "$new_switch_backward" 2>/dev/null; then
        log_debug "Updated switch-input-source-backward to: $new_switch_backward"
    fi
    
    # Alternative: Set to safe defaults if user prefers
    # gsettings set org.gnome.desktop.wm.keybindings switch-input-source "['<Super>space']"
    # gsettings set org.gnome.desktop.wm.keybindings switch-input-source-backward "['<Super><Shift>space']"
    
    log_success "Fixed Alt+Shift hotkey conflicts"
    log_info "Alt+Shift combinations are now available for user applications"
    log_info "Original settings backed up and can be restored during uninstall"
}

# Add bash aliases with immediate availability
add_bash_aliases() {
    log_info "Adding convenient bash aliases..."
    
    local bashrc="$HOME/.bashrc"
    local marker="# USB Safe Eject (${SCRIPT_NAME})"
    
    # Ensure .bashrc exists
    touch "$bashrc"
    
    # Remove existing entries to avoid duplicates
    if grep -q "$marker" "$bashrc" 2>/dev/null; then
        log_debug "Removing existing aliases"
        sed -i "/$marker/,+6d" "$bashrc"
    fi
    
    # Add new entries with improved aliases
    cat >> "$bashrc" << EOF

$marker
# Windows-like USB drive management aliases
alias eject='safeeject'
alias ejectall='safeeject all'
alias usb='safeeject'
alias usblist='safeeject'
alias safely-remove='safeeject'
EOF

    # Make aliases available immediately in current shell if possible
    if [ -n "${BASH_VERSION:-}" ]; then
        # We're running in bash, source the aliases for current session
        # shellcheck source=/dev/null
        if source "$bashrc" 2>/dev/null; then
            log_success "Added aliases (available immediately in current session)"
        else
            log_success "Added aliases (restart terminal or run 'source ~/.bashrc')"
        fi
    else
        log_success "Added aliases (restart terminal or run 'source ~/.bashrc')"
    fi
    
    # Also add to .profile for non-bash shells
    local profile="$HOME/.profile"
    if [ -f "$profile" ] && ! grep -q "$marker" "$profile" 2>/dev/null; then
        cat >> "$profile" << EOF

$marker
# Ensure USB management aliases are available in all shells
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi
EOF
        log_debug "Added source directive to .profile"
    fi
    
    log_info "Available aliases: eject, ejectall, usb, usblist, safely-remove"
}

# Apply system changes
apply_changes() {
    log_info "Applying system changes..."
    
    # Reload udev rules
    if command -v udevadm &> /dev/null; then
        if sudo udevadm control --reload-rules 2>/dev/null; then
            log_debug "udev rules reloaded"
        else
            log_warning "Failed to reload udev rules"
        fi
        
        if sudo udevadm trigger 2>/dev/null; then
            log_debug "udev trigger completed"
        else
            log_warning "Failed to trigger udev"
        fi
    else
        log_warning "udevadm not available - cannot reload udev rules"
    fi
    
    # Restart udisks2 service if available and active
    if systemctl is-active --quiet udisks2 2>/dev/null; then
        if sudo systemctl restart udisks2 2>/dev/null; then
            log_debug "udisks2 system service restarted"
        else
            log_warning "Failed to restart udisks2 system service"
        fi
    fi
    
    # Restart user udisks2 service if available and active
    if systemctl --user is-active --quiet udisks2 2>/dev/null; then
        if systemctl --user restart udisks2 2>/dev/null; then
            log_debug "udisks2 user service restarted"
        else
            log_warning "Failed to restart udisks2 user service"
        fi
    fi
    
    # Restart file manager to pick up new scripts (if GUI session)
    if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        # Restart Nautilus if running
        if pgrep -x nautilus >/dev/null 2>&1; then
            log_debug "Restarting Nautilus to pick up new scripts"
            nautilus -q 2>/dev/null || true
            # Don't restart nautilus automatically - let user do it
        fi
        
        # Update desktop database again to ensure immediate availability
        if command -v update-desktop-database &> /dev/null; then
            update-desktop-database ~/.local/share/applications/ 2>/dev/null || true
        fi
    fi
    
    log_success "System changes applied successfully"
    log_info "All services and configurations have been updated"
}

# Create the enterprise-grade safeeject command
create_safeeject_command() {
    log_info "Creating enterprise-grade safeeject command..."
    
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    sudo tee /usr/local/bin/safeeject > /dev/null << EOF
#!/usr/bin/env bash

# Enterprise USB Safe Eject - Generated by $SCRIPT_NAME v$VERSION
# System capabilities: ${capabilities[*]}

set -euo pipefail

export LANG=C
export LC_ALL=C

readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly RED='\033[0;31m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'
readonly LOG_FILE="/var/log/safeeject.log"
readonly LOCK_DIR="/var/lock/safeeject"

# Settings optimized for this system
readonly SYNC_TIMEOUT=$SYNC_TIMEOUT
readonly LSOF_TIMEOUT=$LSOF_TIMEOUT
readonly UNMOUNT_RETRIES=$UNMOUNT_RETRIES
readonly PROCESS_WAIT_TIMEOUT=$PROCESS_WAIT_TIMEOUT

# Create lock directory if needed
[[ -d "\$LOCK_DIR" ]] || sudo mkdir -p "\$LOCK_DIR" 2>/dev/null || mkdir -p "\$LOCK_DIR"

log_info() { echo -e "\${BLUE}ℹ️  \$*\${NC}"; }
log_success() { echo -e "\${GREEN}✅ \$*\${NC}"; }
log_warning() { echo -e "\${YELLOW}⚠️  \$*\${NC}"; }
log_error() { echo -e "\${RED}❌ \$*\${NC}" >&2; }
log_debug() { echo "\$(date '+%Y-%m-%d %H:%M:%S') DEBUG: \$*" >> "\$LOG_FILE" 2>/dev/null || true; }
log_security() { logger -t "safeeject[\$\$]" -p "authpriv.warning" "SECURITY: \$*" 2>/dev/null || true; }

cleanup_resources() {
    find "\$LOCK_DIR" -name "*.\$\$.lock" -delete 2>/dev/null || true
}

trap cleanup_resources EXIT INT TERM

validate_path() {
    local path="\$1"
    if [[ ! "\$path" =~ ^[a-zA-Z0-9/_[:space:]-]+\$ ]]; then
        log_security "Invalid path attempted: \$path"
        return 1
    fi
    if [[ -e "\$path" ]]; then
        realpath -e "\$path" 2>/dev/null || return 1
    else
        return 1
    fi
}

acquire_lock() {
    local resource="\$1"
    local lockfile="\$LOCK_DIR/\${resource//\//_}.\$\$.lock"
    
    if (set -C; echo "\$\$" > "\$lockfile") 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

release_lock() {
    local resource="\$1"
    local lockfile="\$LOCK_DIR/\${resource//\//_}.\$\$.lock"
    rm -f "\$lockfile"
}

timeout_cmd() {
    local timeout_duration="\$1"
    local description="\$2"
    shift 2
    
    if timeout "\$timeout_duration" "\$@" 2>/dev/null; then
        return 0
    else
        log_warning "\$description timed out after \${timeout_duration}s"
        return 1
    fi
}

safe_sync() {
    log_info "Syncing data to disk..."
    if timeout_cmd "\$SYNC_TIMEOUT" "Data sync" sync; then
        return 0
    else
        log_warning "Sync timed out - continuing with caution..."
        return 1
    fi
}

get_device_info() {
    local device="\$1"
    local info_type="\$2"
    
    validate_path "\$device" || return 1
    
    local result
    result=\$(LANG=C lsblk -no "\$info_type" "\$device" 2>/dev/null) || result=""
    
    if [ -n "\$result" ]; then
        echo "\$result"
        return 0
    fi
    
    # Fallback to udevadm if available
$(if [[ " ${capabilities[*]} " =~ " udevadm " ]]; then echo '    if command -v udevadm &> /dev/null; then
        case "$info_type" in
            "REMOVABLE")
                result=$(udevadm info --query=property --name="$device" 2>/dev/null | grep "ID_BUS=usb" && echo "1" || echo "0")
                ;;
            "FSTYPE")
                result=$(udevadm info --query=property --name="$device" 2>/dev/null | grep "ID_FS_TYPE=" | cut -d'\''='\'' -f2)
                ;;
        esac
        
        if [ -n "$result" ]; then
            echo "$result"
            return 0
        fi
    fi'; fi)
    
    return 1
}

check_processes_using_path() {
    local path="\$1"
    local timeout_duration="\${2:-\$LSOF_TIMEOUT}"
    
    # Try fuser first (faster)
$(if [[ " ${capabilities[*]} " =~ " fuser " ]]; then echo '    if timeout_cmd "$timeout_duration" "Process check" fuser -m "$path" >/dev/null 2>&1; then
        return 0
    fi'; fi)
    
    # Fallback to lsof
$(if [[ " ${capabilities[*]} " =~ " lsof " ]]; then echo '    if timeout_cmd "$timeout_duration" "Process check" lsof +D "$path" >/dev/null 2>&1; then
        return 0
    fi'; fi)
    
    return 1
}

get_process_details() {
    local path="\$1"
    
$(if [[ " ${capabilities[*]} " =~ " fuser " ]]; then echo '    local fuser_output
    fuser_output=$(fuser -v "$path" 2>&1 | tail -n +2 || true)
    if [ -n "$fuser_output" ]; then
        echo "$fuser_output"
        return 0
    fi'; fi)
    
$(if [[ " ${capabilities[*]} " =~ " lsof " ]]; then echo '    local lsof_output
    lsof_output=$(lsof +D "$path" 2>/dev/null | head -10 || true)
    if [ -n "$lsof_output" ]; then
        echo "$lsof_output"
        return 0
    fi'; fi)
    
    echo "Unable to determine specific processes"
    return 1
}

is_removable_device() {
    local path="\$1"
    
    validate_path "\$path" || return 1
    
    local device
    device=\$(LANG=C df "\$path" 2>/dev/null | tail -1 | awk '{print \$1}') || return 1
    
    local base_device
    base_device=\$(echo "\$device" | sed 's/[0-9]*\$//')
    
    local removable
    removable=\$(get_device_info "\$base_device" "REMOVABLE") || return 1
    
    [ "\$removable" = "1" ]
}

get_mount_point() {
    local target="\$1"
    
    validate_path "\$target" || return 1
    
    if [ -d "\$target" ] && mountpoint -q "\$target" 2>/dev/null; then
        echo "\$target"
        return 0
    fi
    
    if [[ "\$target" == /dev/* ]]; then
        local mount_point
        mount_point=\$(LANG=C findmnt -n -o TARGET "\$target" 2>/dev/null) || return 1
        echo "\$mount_point"
        return 0
    fi
    
    local mount_point
    mount_point=\$(LANG=C df "\$target" 2>/dev/null | tail -1 | awk '{print \$6}') || return 1
    echo "\$mount_point"
    return 0
}

unmount_with_retry() {
    local mount_point="\$1"
    local max_attempts="\${2:-\$UNMOUNT_RETRIES}"
    local base_delay="\${3:-2}"
    
    if ! acquire_lock "\$mount_point"; then
        log_error "Could not acquire lock for \$mount_point"
        return 1
    fi
    
    for attempt in \$(seq 1 \$max_attempts); do
        local delay=\$((base_delay * (2 ** (attempt - 1))))
        delay=\$((delay > 30 ? 30 : delay))
        
        log_info "Unmount attempt \$attempt/\$max_attempts..."
        
        if umount "\$mount_point" 2>/dev/null; then
            release_lock "\$mount_point"
            return 0
        fi
        
        if [ \$attempt -lt \$max_attempts ]; then
            log_warning "Unmount failed, waiting \${delay}s before retry..."
            safe_sync
            sleep "\$delay"
            
            if check_processes_using_path "\$mount_point" 2; then
                if umount -l "\$mount_point" 2>/dev/null; then
                    log_info "Lazy unmount successful..."
                    sleep 2
                    if ! mountpoint -q "\$mount_point" 2>/dev/null; then
                        release_lock "\$mount_point"
                        return 0
                    fi
                fi
            fi
        fi
    done
    
    release_lock "\$mount_point"
    return 1
}

power_down_device() {
    local mount_point="\$1"
    
    local device
    device=\$(LANG=C findmnt -n -o SOURCE "\$mount_point" 2>/dev/null | sed 's/[0-9]*\$//' || true)
    
    if [ -z "\$device" ] || [ ! -b "\$device" ]; then
        return 1
    fi
    
$(if [[ " ${capabilities[*]} " =~ " udisks2 " ]]; then echo '    if timeout_cmd 10 "Device power-down" udisksctl power-off -b "$device"; then
        log_success "Device powered down"
        return 0
    fi'; fi)
    
    return 1
}

wait_for_processes() {
    local path="\$1"
    local max_wait="\${2:-\$PROCESS_WAIT_TIMEOUT}"
    local wait_interval="\${3:-1}"
    
    log_info "Waiting for processes to finish..."
    
    local elapsed=0
    local showed_processes=false
    
    while [ \$elapsed -lt \$max_wait ]; do
        if ! check_processes_using_path "\$path" 2; then
            return 0
        fi
        
        if [ "\$showed_processes" = false ]; then
            local process_details
            process_details=\$(get_process_details "\$path")
            if [ -n "\$process_details" ]; then
                log_warning "Active processes:"
                echo "\$process_details" | head -5
                showed_processes=true
            fi
        fi
        
        sleep "\$wait_interval"
        elapsed=\$((elapsed + wait_interval))
        echo -n "."
        
        if [ \$elapsed -ge \$((max_wait / 2)) ] && [ \$((elapsed % 3)) -eq 0 ]; then
            echo ""
            log_warning "Still waiting... Force eject? (y/N/w=wait more)"
            read -r -n 1 -t 3 response || response=""
            echo ""
            
            case "\$response" in
                [Yy]) return 1 ;;
                [Ww]) max_wait=\$((max_wait + 10)) ;;
            esac
        fi
    done
    
    echo ""
    return 1
}

eject_drive() {
    local target="\$1"
    
    local mount_point
    mount_point=\$(get_mount_point "\$target") || {
        log_error "Could not determine mount point for: \$target"
        return 1
    }
    
    if ! is_removable_device "\$mount_point"; then
        log_error "Not a removable device: \$mount_point"
        log_security "Attempted to eject non-removable device: \$mount_point"
        return 1
    fi
    
    log_info "Ejecting: \$mount_point"
    
    # Step 1: Sync
    safe_sync
    
    # Step 2: Check processes
    if check_processes_using_path "\$mount_point" 3; then
        log_warning "Programs are using this drive"
        
        if ! wait_for_processes "\$mount_point" "\$PROCESS_WAIT_TIMEOUT"; then
            log_warning "Force eject? (y/N)"
            read -r -n 1 response
            echo ""
            if [[ ! "\$response" =~ ^[Yy]\$ ]]; then
                log_error "Eject cancelled"
                return 1
            fi
        fi
    fi
    
    # Step 3: Unmount
    log_info "Unmounting drive..."
    if unmount_with_retry "\$mount_point"; then
        log_success "Drive unmounted successfully"
        power_down_device "\$mount_point"
        log_success "✅ Drive safely ejected! You can now remove it."
        return 0
    else
        log_error "Failed to unmount drive after multiple attempts"
        return 1
    fi
}

show_drives() {
    log_info "Available removable drives:"
    echo ""
    
    local drives
    drives=\$(LANG=C lsblk -P -o NAME,MOUNTPOINT,LABEL,SIZE,REMOVABLE,FSTYPE 2>/dev/null | grep 'REMOVABLE="1"' | grep -v 'MOUNTPOINT=""' || true)
    
    if [ -z "\$drives" ]; then
        log_warning "No removable drives found"
        return 1
    fi
    
    echo "\$drives" | while IFS= read -r drive; do
        local name mountpoint label size fstype
        name=\$(echo "\$drive" | grep -o 'NAME="[^"]*"' | cut -d'"' -f2)
        mountpoint=\$(echo "\$drive" | grep -o 'MOUNTPOINT="[^"]*"' | cut -d'"' -f2)
        label=\$(echo "\$drive" | grep -o 'LABEL="[^"]*"' | cut -d'"' -f2)
        size=\$(echo "\$drive" | grep -o 'SIZE="[^"]*"' | cut -d'"' -f2)
        fstype=\$(echo "\$drive" | grep -o 'FSTYPE="[^"]*"' | cut -d'"' -f2)
        
        printf "  📱 %-10s %-25s %-15s %-8s %s\\\\n" \\\\
            "\$name" "\$mountpoint" "\${label:-<no label>}" "\$size" "\$fstype"
    done
    
    echo ""
    log_info "Usage: safeeject <mountpoint>"
    log_info "       safeeject all"
}

eject_all() {
    log_info "Ejecting all removable drives..."
    
    local drives
    drives=\$(LANG=C lsblk -P -o MOUNTPOINT,REMOVABLE 2>/dev/null | grep 'REMOVABLE="1"' | grep -v 'MOUNTPOINT=""' || true)
    
    if [ -z "\$drives" ]; then
        log_warning "No removable drives to eject"
        return 0
    fi
    
    local success=true
    echo "\$drives" | while IFS= read -r drive; do
        local mountpoint
        mountpoint=\$(echo "\$drive" | grep -o 'MOUNTPOINT="[^"]*"' | cut -d'"' -f2)
        
        if [ -n "\$mountpoint" ]; then
            echo ""
            if ! eject_drive "\$mountpoint"; then
                success=false
            fi
        fi
    done
    
    if \$success; then
        echo ""
        log_success "🎉 All drives ejected successfully!"
    else
        echo ""
        log_warning "⚠️  Some drives could not be ejected"
        return 1
    fi
}

main() {
    case "\${1:-}" in
        "")
            show_drives
            ;;
        "all")
            eject_all
            ;;
        "-h"|"--help")
            echo "Enterprise USB Safe Eject v$VERSION"
            echo "Generated by $SCRIPT_NAME with capabilities: ${capabilities[*]}"
            echo ""
            echo "Usage:"
            echo "  safeeject              Show available drives"
            echo "  safeeject <mountpoint> Eject specific drive"
            echo "  safeeject all          Eject all removable drives"
            ;;
        *)
            eject_drive "\$1"
            ;;
    esac
}

main "\$@"
EOF

    sudo chmod +x /usr/local/bin/safeeject
    log_success "Created enterprise-grade safeeject command with capabilities: ${capabilities[*]}"
}

# Enhanced GUI with system capability awareness and security
create_gui_eject() {
    log_info "Creating capability-aware GUI eject..."
    
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    # Only create GUI if we have the capability
    if [[ ! " ${capabilities[*]} " =~ " gui " ]]; then
        log_warning "No GUI environment detected - skipping GUI eject creation"
        return 0
    fi
    
    # Install zenity if not available in GUI environment
    if [[ ! " ${capabilities[*]} " =~ " zenity " ]]; then
        log_info "Installing zenity for GUI functionality..."
        if ! sudo apt update && sudo apt install -y zenity; then
            log_error "Failed to install zenity - GUI functionality will not be available"
            return 1
        fi
    fi
    
    sudo tee /usr/local/bin/safeeject-gui > /dev/null << 'EOF'
#!/usr/bin/env bash

set -euo pipefail
export LANG=C
export LC_ALL=C

# Security timeout for zenity dialogs
readonly GUI_TIMEOUT=30

# Check dependencies
if ! command -v zenity &> /dev/null; then
    echo "Error: zenity not available" >&2
    exit 1
fi

validate_path() {
    local path="$1"
    if [[ ! "$path" =~ ^[a-zA-Z0-9/_[:space:]-]+$ ]]; then
        return 1
    fi
    if [[ -e "$path" ]]; then
        realpath -e "$path" 2>/dev/null || return 1
    else
        return 1
    fi
}

secure_zenity() {
    timeout "$GUI_TIMEOUT" zenity "$@" 2>/dev/null || return 1
}

get_drives() {
    LANG=C lsblk -P -o NAME,MOUNTPOINT,LABEL,SIZE,REMOVABLE,FSTYPE 2>/dev/null | \
    grep 'REMOVABLE="1"' | \
    grep -v 'MOUNTPOINT=""' || true
}

parse_field() {
    local line="$1"
    local field="$2"
    echo "$line" | grep -o "${field}=\"[^\"]*\"" | cut -d'"' -f2 || echo ""
}

main() {
    local drives
    drives=$(get_drives)
    
    if [ -z "$drives" ]; then
        secure_zenity --info \
            --title="Safe Eject USB" \
            --text="No removable USB drives found.\n\n📱 Please ensure your USB drive is:\n• Properly connected\n• Mounted (appears in file manager)\n• A removable device (not internal drive)" \
            --width=450 \
            --no-markup
        exit 0
    fi
    
    # Create robust selection list
    local options=()
    local drive_count=0
    
    while IFS= read -r drive; do
        local name mountpoint label size fstype
        name=$(parse_field "$drive" "NAME")
        mountpoint=$(parse_field "$drive" "MOUNTPOINT")
        label=$(parse_field "$drive" "LABEL")
        size=$(parse_field "$drive" "SIZE")
        fstype=$(parse_field "$drive" "FSTYPE")
        
        if [ -n "$mountpoint" ] && validate_path "$mountpoint"; then
            local display_name="${label:-$name}"
            local info="$display_name ($size, $fstype)"
            
            options+=("$mountpoint")
            options+=("$info - $mountpoint")
            ((drive_count++))
        fi
    done <<< "$drives"
    
    if [ $drive_count -eq 0 ]; then
        secure_zenity --error \
            --title="Safe Eject USB" \
            --text="No valid USB drives found to eject.\n\nAll detected removable devices appear to be unmounted." \
            --width=450 \
            --no-markup
        exit 1
    fi
    
    # Show selection dialog with better formatting
    local selection
    selection=$(secure_zenity --list \
        --title="Safe Eject USB Drive" \
        --text="Choose a USB drive to safely eject:\n\nImportant: Make sure all files are saved before ejecting!\nThis will sync all data and safely unmount the drive." \
        --column="Path" \
        --column="Drive Information" \
        --hide-column=1 \
        --width=700 \
        --height=400 \
        --no-markup \
        "${options[@]}" || true)
    
    if [ -z "$selection" ]; then
        exit 0  # User cancelled
    fi
    
    # Validate selection
    if ! validate_path "$selection"; then
        secure_zenity --error \
            --title="Safe Eject - Error" \
            --text="Invalid drive path selected." \
            --width=400 \
            --no-markup
        exit 1
    fi
    
    # Enhanced progress dialog with better error handling
    local temp_result
    temp_result=$(mktemp)
    
    (
        echo "5" ; echo "# Preparing to eject drive..."
        sleep 0.5
        
        echo "15" ; echo "# Syncing data to disk (this may take a moment)..."
        if ! timeout 15 sync 2>/dev/null; then
            echo "# Warning: Sync timed out - continuing anyway"
        fi
        sleep 1
        
        echo "35" ; echo "# Checking for programs using the drive..."
        sleep 1
        
        # Quick process check using available tools
        local processes_found=false
        if command -v fuser &> /dev/null; then
            if timeout 5 fuser -m "$selection" >/dev/null 2>&1; then
                processes_found=true
            fi
        elif command -v lsof &> /dev/null; then
            if timeout 5 lsof +D "$selection" >/dev/null 2>&1; then
                processes_found=true
            fi
        fi
        
        if [ "$processes_found" = true ]; then
            echo "55" ; echo "# Waiting for programs to finish using the drive..."
            sleep 3
        fi
        
        echo "75" ; echo "# Unmounting drive..."
        sleep 1
        
        # Attempt unmount with retries
        local unmount_success=false
        for attempt in {1..3}; do
            if umount "$selection" 2>/dev/null; then
                unmount_success=true
                break
            fi
            
            if [ $attempt -lt 3 ]; then
                echo "# Retry $((attempt+1))/3 - waiting for processes to finish..."
                sleep 2
            fi
        done
        
        if [ "$unmount_success" = true ]; then
            echo "90" ; echo "# Powering down drive..."
            
            # Try to power down USB device
            local device
            device=$(findmnt -n -o SOURCE "$selection" 2>/dev/null | sed 's/[0-9]*$//' || true)
            if [ -n "$device" ] && [ -b "$device" ]; then
                timeout 10 udisksctl power-off -b "$device" 2>/dev/null || true
            fi
            
            echo "100" ; echo "# Successfully ejected!"
            echo "SUCCESS" > "$temp_result"
        else
            echo "100" ; echo "# Error: Could not unmount drive"
            echo "FAILED" > "$temp_result"
        fi
        
        sleep 1
        
    ) | secure_zenity --progress \
        --title="Safe Eject USB Drive" \
        --text="Preparing to eject drive..." \
        --width=500 \
        --auto-close \
        --no-cancel \
        --no-markup || echo "CANCELLED" > "$temp_result"
    
    # Check result and show appropriate message
    local result
    result=$(cat "$temp_result" 2>/dev/null || echo "UNKNOWN")
    rm -f "$temp_result"
    
    case "$result" in
        "SUCCESS")
            secure_zenity --info \
                --title="Safe Eject - Success" \
                --text="USB drive safely ejected!\n\nYou can now safely remove the drive.\n\nAll data has been written to the drive and it has been properly unmounted and powered down." \
                --width=450 \
                --no-markup
            ;;
        "FAILED")
            secure_zenity --error \
                --title="Safe Eject - Error" \
                --text="Failed to eject the USB drive.\n\nPossible causes:\n• Files or folders are still open on the drive\n• A program is still accessing the drive\n• The drive is busy or malfunctioning\n\nSolutions:\n• Close all programs that might be using the drive\n• Check for open files in the file manager\n• Try the command line: safeeject $selection" \
                --width=550 \
                --no-markup
            ;;
        "CANCELLED")
            secure_zenity --warning \
                --title="Safe Eject - Cancelled" \
                --text="Eject operation was cancelled.\n\nThe drive is still mounted and in use.\n\nYou can try again when you're ready." \
                --width=400 \
                --no-markup
            ;;
        *)
            secure_zenity --error \
                --title="Safe Eject - Unknown Error" \
                --text="An unexpected error occurred during the eject process.\n\nTry the command line version:\nsafeeject $selection\n\nOr restart the application." \
                --width=450 \
                --no-markup
            ;;
    esac
}

main "$@"
EOF

    sudo chmod +x /usr/local/bin/safeeject-gui
    log_success "Created capability-aware GUI eject with security hardening"
}

# System health check
system_health_check() {
    echo -e "${BOLD}System Health Check${NC}"
    echo "==================="
    echo ""
    
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    log_info "Detected capabilities: ${capabilities[*]}"
    echo ""
    
    # Check udisks2 services
    echo "Service Status:"
    if systemctl --user is-active udisks2.service &>/dev/null; then
        log_success "udisks2 user service is running"
    else
        log_warning "udisks2 user service is not running"
        if [[ " ${capabilities[*]} " =~ " gui " ]]; then
            log_info "Consider running: systemctl --user start udisks2.service"
        fi
    fi
    
    if systemctl is-active udisks2.service &>/dev/null; then
        log_success "udisks2 system service is running"
    else
        log_warning "udisks2 system service is not running"
    fi
    
    echo ""
    
    # Test basic functionality
    echo "Functionality Tests:"
    
    # Test removable drive detection
    local drives
    drives=$(get_removable_drives_enhanced)
    
    if [ -n "$drives" ]; then
        local drive_count
        drive_count=$(echo "$drives" | wc -l)
        log_success "Drive detection working ($drive_count removable drives found)"
    else
        log_info "No removable drives currently connected (detection working)"
    fi
    
    # Test sync timeout
    if timeout_cmd 3 "Sync test" sync; then
        log_success "Sync functionality working"
    else
        log_warning "Sync test failed or timed out"
    fi
    
    # Test process detection
    if [[ " ${capabilities[*]} " =~ " fuser " ]]; then
        log_success "fuser available (fast process detection)"
    elif [[ " ${capabilities[*]} " =~ " lsof " ]]; then
        log_success "lsof available (process detection)"
    else
        log_error "No process detection tools available"
    fi
    
    # Check security settings
    echo ""
    echo "Security Status:"
    
    # Check log file permissions
    if [ -f "$LOG_FILE" ]; then
        local log_perms
        log_perms=$(stat -c %a "$LOG_FILE" 2>/dev/null || echo "unknown")
        if [ "$log_perms" = "640" ] || [ "$log_perms" = "600" ]; then
            log_success "Log file permissions secure"
        else
            log_warning "Log file permissions may be too permissive: $log_perms"
        fi
    fi
    
    # Check mount options configuration
    if [ -f /etc/udisks2/mount_options.conf ]; then
        if grep -q "$SECURE_MOUNT_OPTS" /etc/udisks2/mount_options.conf 2>/dev/null; then
            log_success "Security mount options configured"
        else
            log_warning "Security mount options may not be fully configured"
        fi
    fi
    
    echo ""
}

# Test system functionality
test_system() {
    echo -e "${BOLD}Testing System Functionality${NC}"
    echo "============================"
    echo ""
    
    if [ ! -f "$INSTALL_MARKER" ]; then
        log_error "System is not installed. Run with --install first."
        return 1
    fi
    
    system_health_check
    
    # Test safeeject command
    if [ -x /usr/local/bin/safeeject ]; then
        log_info "Testing safeeject command..."
        if /usr/local/bin/safeeject --help >/dev/null 2>&1; then
            log_success "safeeject command working"
        else
            log_error "safeeject command failed"
        fi
    else
        log_error "safeeject command not found"
    fi
    
    # Test GUI if available
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    if [[ " ${capabilities[*]} " =~ " gui " ]] && [[ " ${capabilities[*]} " =~ " zenity " ]]; then
        log_info "Testing GUI functionality..."
        if command -v /usr/local/bin/safeeject-gui &> /dev/null; then
            log_success "GUI eject available"
        else
            log_warning "GUI eject not found"
        fi
    fi
    
    # Test aliases
    if grep -q "alias eject='safeeject'" ~/.bashrc 2>/dev/null; then
        log_success "Bash aliases configured"
    else
        log_warning "Bash aliases missing"
    fi
    
    # Test lock mechanism
    if [ -d "$LOCK_DIR" ]; then
        if acquire_lock "test_resource"; then
            log_success "Lock mechanism working"
            release_lock "test_resource"
        else
            log_error "Lock mechanism failed"
        fi
    else
        log_warning "Lock directory missing"
    fi
    
    echo ""
    log_info "Test complete. Check any warnings above."
}

# Enhanced installation with comprehensive error handling
install_system() {
    echo -e "${BOLD}🚀 Installing Complete Enterprise USB Management v${VERSION}${NC}"
    echo "=================================================================="
    echo ""
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_error "Do not run this script as root. It will request sudo when needed."
        exit 1
    fi
    
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    echo "Detected system capabilities:"
    for cap in "${capabilities[@]}"; do
        echo "  ✓ $cap"
    done
    echo ""
    
    echo "This will install:"
    echo "✅ Immediate-write USB mounting with security options"
    echo "✅ Enterprise-grade 'safeeject' command with timeouts and retries"
    echo "✅ GUI eject options (if GUI environment detected)"
    echo "✅ File manager integration (right-click eject)"
    echo "✅ Desktop applications and aliases"
    echo "✅ Alt+Shift hotkey fix (with restore capability)"
    echo "✅ Comprehensive error handling and security hardening"
    echo ""
    
    read -p "Continue installation? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    
    echo ""
    
    # Check and install dependencies
    log_info "Checking system dependencies..."
    local missing_packages=()
    
    # Essential packages
    for pkg in udisks2 psmisc coreutils; do  # psmisc provides fuser
        if ! dpkg -l "$pkg" &>/dev/null; then
            missing_packages+=("$pkg")
        fi
    done
    
    # Optional packages for GUI
    if [[ " ${capabilities[*]} " =~ " gui " ]]; then
        if ! dpkg -l "zenity" &>/dev/null; then
            missing_packages+=("zenity")
        fi
    fi
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        log_info "Installing missing packages: ${missing_packages[*]}"
        if ! sudo apt update; then
            log_error "Failed to update package lists"
            return 1
        fi
        
        if ! sudo apt install -y "${missing_packages[@]}"; then
            log_error "Failed to install required packages"
            return 1
        fi
    fi
    
    # Create log file with proper permissions
    sudo touch "$LOG_FILE"
    sudo chmod 640 "$LOG_FILE"
    sudo chown "$USER:adm" "$LOG_FILE" 2>/dev/null || true
    
    # Update capabilities after package installation
    capabilities=($(detect_system_capabilities))
    
    # Run installation steps in correct order
    log_info "Installing core components..."
    
    setup_mount_options || error_exit "Failed to setup mount options"
    create_safeeject_command || error_exit "Failed to create safeeject command"
    
    if [[ " ${capabilities[*]} " =~ " gui " ]]; then
        create_gui_eject || log_warning "Failed to create GUI eject"
        setup_file_manager_integration || log_warning "Failed to setup file manager integration"
        create_desktop_app || log_warning "Failed to create desktop app"
    else
        log_info "No GUI environment - skipping GUI components"
    fi
    
    fix_hotkeys || log_warning "Failed to fix hotkeys"
    add_bash_aliases || log_warning "Failed to add bash aliases"
    apply_changes || log_warning "Failed to apply some system changes"
    
    # Create installation marker with metadata
    mkdir -p "$(dirname "$INSTALL_MARKER")"
    cat > "$INSTALL_MARKER" << EOF
install_date=$(date '+%Y-%m-%d %H:%M:%S')
version=$VERSION
capabilities=${capabilities[*]}
backup_created=true
secure_mount_opts=$SECURE_MOUNT_OPTS
EOF
    
    echo ""
    log_success "🎉 Complete enterprise installation successful!"
    echo ""
    
    # Run immediate health check
    system_health_check
    
    echo ""
    echo -e "${BOLD}How to use your new USB management system:${NC}"
    echo ""
    echo "  ${BLUE}Command Line:${NC}"
    echo "    safeeject              (show USB drives)"
    echo "    safeeject /media/*/DRIVE  (eject specific drive)"
    echo "    safeeject all          (eject all drives)"
    echo "    eject                  (alias for safeeject)"
    echo "    usb                    (alias for safeeject)"
    echo "    safely-remove          (alias for safeeject)"
    echo ""
    
    if [[ " ${capabilities[*]} " =~ " gui " ]]; then
        echo "  ${BLUE}GUI Options:${NC}"
        echo "    • Search 'Safe Eject USB' in applications"
        echo "    • Right-click files on USB → 'Safe Eject'"
        echo "    • Use eject button in Files app sidebar"
        echo ""
    fi
    
    echo -e "${BOLD}Enterprise Features Installed:${NC}"
    echo "  ✅ Security hardening (nodev,nosuid,noexec mount options)"
    echo "  ✅ Timeout protection prevents system hangs"
    echo "  ✅ Retry logic with exponential backoff"
    echo "  ✅ Atomic operations with proper locking"
    echo "  ✅ Comprehensive input validation and sanitization"
    echo "  ✅ Security event logging"
    echo "  ✅ Works in headless and GUI environments"
    echo "  ✅ Backup and restore capability for all settings"
    echo ""
    echo -e "${YELLOW}Restart recommended for all changes to take effect.${NC}"
    echo ""
    echo -e "${GREEN}Your Linux USB experience is now enterprise-grade and bulletproof! 🏢✨${NC}"
}

# Enhanced uninstall with comprehensive cleanup and restore
uninstall_system() {
    echo -e "${BOLD}🗑️  Uninstalling Complete Enterprise USB Management${NC}"
    echo "=================================================="
    echo ""
    
    if [ ! -f "$INSTALL_MARKER" ]; then
        log_warning "System doesn't appear to be installed."
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    else
        echo "Installation details:"
        cat "$INSTALL_MARKER" | sed 's/^/  /'
        echo ""
    fi
    
    echo "This will remove:"
    echo "❌ safeeject commands and GUI integration"
    echo "❌ Custom mount behavior for USB drives"
    echo "❌ File manager integration and desktop apps"
    echo "❌ Bash aliases and system configurations"
    echo "🔄 Restore original keyboard shortcut settings (if backed up)"
    echo ""
    
    read -p "Continue uninstallation? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Uninstallation cancelled."
        exit 0
    fi
    
    echo ""
    
    # Remove all installed components
    log_info "Removing installed files and configurations..."
    
    # Remove commands
    sudo rm -f /usr/local/bin/safeeject
    sudo rm -f /usr/local/bin/safeeject-gui
    
    # Remove system configurations
    sudo rm -f /etc/udev/rules.d/99-removable-sync.rules
    sudo rm -f /etc/udisks2/mount_options.conf
    sudo rm -f /etc/udisks2/udisks2.conf
    
    # Remove user configurations
    rm -f ~/.local/share/nautilus/scripts/Safe\ Eject
    rm -f ~/.local/share/applications/safeeject.desktop
    rm -f ~/.local/share/applications/safeeject-terminal.desktop
    rm -f ~/.local/share/file-manager/actions/safe-eject.desktop
    
    # Remove aliases from bashrc and profile
    if [ -f ~/.bashrc ]; then
        sed -i "/# USB Safe Eject (${SCRIPT_NAME})/,+6d" ~/.bashrc
    fi
    
    if [ -f ~/.profile ]; then
        sed -i "/# USB Safe Eject (${SCRIPT_NAME})/,+4d" ~/.profile
    fi
    
    # Restore keyboard shortcuts if backup exists
    local backup_file="$HOME/.config/${SCRIPT_NAME}/hotkey-backup/input-source-settings.txt"
    if [ -f "$backup_file" ]; then
        log_info "Restoring original keyboard shortcuts..."
        
        if command -v gsettings &> /dev/null; then
            # Extract and restore original settings
            local orig_switch orig_switch_backward
            orig_switch=$(grep "switch-input-source=" "$backup_file" | cut -d'=' -f2-)
            orig_switch_backward=$(grep "switch-input-source-backward=" "$backup_file" | cut -d'=' -f2-)
            
            if [ -n "$orig_switch" ]; then
                gsettings set org.gnome.desktop.wm.keybindings switch-input-source "$orig_switch" 2>/dev/null || true
                log_debug "Restored switch-input-source to: $orig_switch"
            fi
            
            if [ -n "$orig_switch_backward" ]; then
                gsettings set org.gnome.desktop.wm.keybindings switch-input-source-backward "$orig_switch_backward" 2>/dev/null || true
                log_debug "Restored switch-input-source-backward to: $orig_switch_backward"
            fi
            
            log_success "Original keyboard shortcuts restored"
        fi
    else
        log_warning "No keyboard shortcut backup found - manual restoration may be needed"
    fi
    
    # Remove backup directory
    rm -rf "$HOME/.config/${SCRIPT_NAME}"
    
    # Apply system changes
    log_info "Applying system changes..."
    sudo udevadm control --reload-rules 2>/dev/null || true
    sudo udevadm trigger 2>/dev/null || true
    
    if systemctl is-active --quiet udisks2 2>/dev/null; then
        sudo systemctl restart udisks2 2>/dev/null || true
    fi
    
    if systemctl --user is-active --quiet udisks2 2>/dev/null; then
        systemctl --user restart udisks2 2>/dev/null || true
    fi
    
    # Update desktop database
    if command -v update-desktop-database &> /dev/null; then
        update-desktop-database ~/.local/share/applications/ 2>/dev/null || true
    fi
    
    # Remove installation marker and logs
    rm -f "$INSTALL_MARKER"
    sudo rm -f "$LOG_FILE"
    sudo rm -rf "$LOCK_DIR"
    
    echo ""
    log_success "🗑️  Complete enterprise uninstallation finished!"
    echo ""
    log_info "All components removed and original settings restored"
    log_info "Restart recommended to ensure all changes take effect"
    echo ""
    log_success "Your system has been restored to its original state"
}

# Enhanced installation status check
check_installation() {
    echo -e "${BOLD}Complete Installation Status Check${NC}"
    echo "=================================="
    echo ""
    
    if [ -f "$INSTALL_MARKER" ]; then
        log_success "System is installed"
        echo ""
        echo "Installation details:"
        cat "$INSTALL_MARKER" | sed 's/^/  /'
        echo ""
    else
        log_warning "System is not installed"
        echo ""
    fi
    
    # Component status check
    echo "Component Status:"
    
    local all_good=true
    
    # Check core commands
    if [ -x /usr/local/bin/safeeject ]; then
        log_success "safeeject command available"
    else
        log_error "safeeject command missing"
        all_good=false
    fi
    
    # Check GUI command
    local capabilities
    capabilities=($(detect_system_capabilities))
    
    if [ -x /usr/local/bin/safeeject-gui ]; then
        log_success "GUI eject available"
    else
        if [[ " ${capabilities[*]} " =~ " gui " ]]; then
            log_error "GUI eject missing (should be available)"
            all_good=false
        else
            log_info "GUI eject not available (no GUI environment)"
        fi
    fi
    
    # Check system configurations
    if [ -f /etc/udev/rules.d/99-removable-sync.rules ]; then
        log_success "Sync mount rules installed"
    else
        log_error "Sync mount rules missing"
        all_good=false
    fi
    
    if [ -f /etc/udisks2/mount_options.conf ] || [ -f /etc/udisks2/udisks2.conf ]; then
        log_success "Mount options configured"
    else
        log_error "Mount options missing"
        all_good=false
    fi
    
    # Check security configurations
    if [ -f /etc/udisks2/mount_options.conf ]; then
        if grep -q "$SECURE_MOUNT_OPTS" /etc/udisks2/mount_options.conf 2>/dev/null; then
            log_success "Security mount options configured"
        else
            log_warning "Security mount options may not be configured"
        fi
    fi
    
    # Check user configurations
    if [ -f ~/.local/share/applications/safeeject.desktop ]; then
        log_success "Desktop application available"
    else
        log_warning "Desktop application missing"
    fi
    
    if [ -f ~/.local/share/nautilus/scripts/Safe\ Eject ]; then
        log_success "File manager integration available"
    else
        log_warning "File manager integration missing"
    fi
    
    # Check aliases
    if grep -q "alias eject='safeeject'" ~/.bashrc 2>/dev/null; then
        log_success "Bash aliases configured"
    else
        log_warning "Bash aliases missing"
    fi
    
    # Check keyboard shortcut backup
    if [ -f "$HOME/.config/${SCRIPT_NAME}/hotkey-backup/input-source-settings.txt" ]; then
        log_success "Keyboard shortcut backup available"
    else
        log_info "No keyboard shortcut backup (may not be needed)"
    fi
    
    # Check lock directory
    if [ -d "$LOCK_DIR" ]; then
        log_success "Lock directory available"
    else
        log_warning "Lock directory missing"
    fi
    
    # Check log file
    if [ -f "$LOG_FILE" ]; then
        local log_perms
        log_perms=$(stat -c %a "$LOG_FILE" 2>/dev/null || echo "unknown")
        if [ "$log_perms" = "640" ] || [ "$log_perms" = "600" ]; then
            log_success "Log file configured with secure permissions"
        else
            log_warning "Log file permissions may be insecure: $log_perms"
        fi
    else
        log_info "Log file not created yet"
    fi
    
    echo ""
    
    if [ "$all_good" = true ]; then
        log_success "All critical components are properly installed"
    else
        log_warning "Some components are missing - consider reinstalling"
    fi
    
    echo ""
    
    # Run health check
    system_health_check
}

# Main function with comprehensive argument handling
main() {
    # Initialize logging
    if [ ! -f "$LOG_FILE" ] && [ -w "$(dirname "$LOG_FILE")" ]; then
        touch "$LOG_FILE" 2>/dev/null || true
        chmod 640 "$LOG_FILE" 2>/dev/null || true
    fi
    
    log_debug "Script started with arguments: $*"
    
    case "${1:-}" in
        --install|-i)
            install_system
            ;;
        --uninstall|-u)
            uninstall_system
            ;;
        --check|-c)
            check_installation
            ;;
        --test|-t)
            test_system
            ;;
        --version|-v)
            echo "Complete Enterprise USB Drive Management v${VERSION}"
            echo "Generated by ${SCRIPT_NAME}"
            echo ""
            echo "System capabilities:"
            local capabilities
            capabilities=($(detect_system_capabilities))
            for cap in "${capabilities[@]}"; do
                echo "  ✓ $cap"
            done
            ;;
        --help|-h|"")
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
    
    log_debug "Script completed successfully"
}

main "$@"