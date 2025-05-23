#date: 2025-05-23T16:44:09Z
#url: https://api.github.com/gists/3030fba09c049ccdcb1c7cbeec099543
#owner: https://api.github.com/users/ZeroDot1

#!/bin/bash
#
# SmartDriveCheckNG.sh - Drive Health Report Utility
#
# Copyright (C) 2025 ZeroDot1
# License: GNU Affero General Public License, Version 3 (AGPLv3)
#
# This script scans block devices (HDDs, SSDs, floppies, CDROMs) on Arch Linux,
# lets the user select a device, and generates a detailed SMART/NVMe health report.
# Reports are saved in /tmp and include system info, date, time, debug status,
# and drive health categorized as Healthy, Critical, or Failed.
#
# Usage: ./SmartDriveCheckNG.sh [--debug]
#
# Dependencies: smartctl lsblk numfmt column lscpu free lspci dmidecode yay leafpad nvme
# Author: ZeroDot1 <https://github.com/ZeroDot1>
#

# --- Configuration ---
PROG_NAME="SmartDriveCheckNG"
REPORT_BASE_DIR="/tmp"
_DEBUG=0
SUDO_CMD=""
REQUIRED_CMDS="smartctl lsblk numfmt column lscpu free lspci dmidecode yay leafpad nvme"
TEMP_FILES=()

# --- Bash Strict Mode ---
set -euo pipefail
IFS=$'\n\t'

# --- Cleanup Function ---
_cleanup() {
    [ "$_DEBUG" -eq 1 ] && echo "DEBUG: Cleaning up temporary files..." >&2
    for file in "${TEMP_FILES[@]}"; do
        [ -f "$file" ] && { rm -f "$file" && [ "$_DEBUG" -eq 1 ] && echo "DEBUG: Removed $file" >&2; } || echo "ERROR: Failed to remove $file" >&2
    done
}
trap _cleanup EXIT

# --- Utility Functions ---
_debug() { [ "$_DEBUG" -eq 1 ] && echo "DEBUG: $1" >&2; }
_error() { echo "ERROR: $1" >&2; }
_info() { echo "INFO: $1" >&2; }
_pause() { read -r -p "$1 (Press Enter to continue): " >&2; }

_ask_yes_no() {
    local question="$1" default="${2:-y}" response
    while true; do
        read -r -p "$question ($( [ "$default" = "y" ] && echo "Y/n" || echo "y/N" )): " response
        response="${response:-$default}"
        case "${response,,}" in
            y|yes) return 0 ;;
            n|no) return 1 ;;
            *) echo "Please enter 'y' or 'n'." >&2 ;;
        esac
    done
}

# --- Dependency Management ---
_install_deps() {
    local missing=() cmd pkg
    for cmd in $REQUIRED_CMDS; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done

    if [ ${#missing[@]} -gt 0 ]; then
        _info "Missing dependencies: ${missing[*]}"
        local packages=""
        for cmd in "${missing[@]}"; do
            case "$cmd" in
                smartctl) packages+="smartmontools " ;;
                numfmt|lsblk|column) packages+="util-linux " ;;
                lscpu|free|lspci) packages+="util-linux procps-ng pciutils " ;;
                dmidecode) packages+="dmidecode " ;;
                yay) _error "yay is missing. Install it manually as a non-root user: https://aur.archlinux.org/yay"; return 1 ;;
                leafpad) packages+="leafpad " ;;
                nvme) packages+="nvme-cli " ;;
            esac
        done

        if [ -n "$packages" ]; then
            if [ "$(id -u)" -eq 0 ]; then
                _error "Cannot run yay as root. Please install the following packages as a non-root user:"
                echo "  yay -S $packages" >&2
                echo "Or use pacman: sudo pacman -S $packages" >&2
                _pause "Install the packages and press Enter to continue"
                for cmd in "${missing[@]}"; do
                    command -v "$cmd" &>/dev/null || { _error "$cmd still missing after manual installation attempt."; return 1; }
                done
            else
                if _ask_yes_no "Install missing packages ($packages) with yay?"; then
                    yay -S $packages --noconfirm || { _error "Failed to install packages with yay."; return 1; }
                    for cmd in "${missing[@]}"; do
                        command -v "$cmd" &>/dev/null || { _error "$cmd still missing after installation."; return 1; }
                    done
                else
                    _error "Dependency installation declined. Cannot proceed."
                    return 1
                fi
            fi
        fi
    fi
    _info "All dependencies satisfied."
    return 0
}

_check_deps() {
    command -v yay &>/dev/null || { _error "yay not found. Install it manually as a non-root user: https://aur.archlinux.org/yay"; exit 1; }
    _install_deps || { _error "Dependency check failed."; _pause "Exiting"; exit 1; }
}

# --- Sudo Setup ---
_set_sudo() {
    if [ "$(id -u)" -eq 0 ]; then
        SUDO_CMD=""
        _info "Running as root. No sudo needed for privileged operations."
    else
        SUDO_CMD="sudo"
        command -v sudo &>/dev/null || { _error "sudo not found. Cannot proceed."; _pause "Exiting"; exit 1; }
        _info "Using sudo for privileged operations."
    fi
}

# --- System Information ---
_get_system_info() {
    local info=""
    info+="Kernel: $(uname -r)\n"
    info+="Hostname: $(hostname)\n"
    info+="Uptime: $(uptime -p)\n"
    info+="\n--- CPU ---\n"
    info+="$($SUDO_CMD lscpu | grep -E 'Model name|Architecture|CPU\(s\):' | sed 's/^[ ]*//')\n"
    info+="\n--- Memory ---\n"
    info+="$($SUDO_CMD free -h | grep -E 'Mem:|Swap:')\n"
    info+="\n--- GPU ---\n"
    info+="$($SUDO_CMD lspci | grep -i vga || echo 'N/A')\n"
    info+="\n--- Motherboard/BIOS ---\n"
    info+="Board: $($SUDO_CMD dmidecode -s baseboard-product-name 2>/dev/null || echo 'N/A')\n"
    info+="BIOS: $($SUDO_CMD dmidecode -s bios-version 2>/dev/null || echo 'N/A')\n"
    echo -e "$info"
}

# --- Drive Discovery ---
_discover_drives() {
    DRIVE_PATHS=()
    DRIVE_NAMES=()
    local tmp_file=$(mktemp -p "${TMPDIR:-/tmp}" "lsblk_data_XXXXXX.txt") || { _error "Failed to create temp file."; return 1; }
    TEMP_FILES+=("$tmp_file")

    $SUDO_CMD lsblk -d -o NAME,MODEL,SIZE,TYPE --bytes --noheadings | grep -Ev '^(part|zram|loop)' >"$tmp_file" || {
        _error "lsblk failed. Check permissions or device availability."
        return 1
    }
    _debug "lsblk output saved to $tmp_file"

    while IFS= read -r line; do
        local name=$(echo "$line" | awk '{print $1}')
        local type=$(echo "$line" | awk '{print $NF}')
        local size=$(echo "$line" | awk '{print $(NF-1)}')
        local model=$(echo "$line" | awk '{$1=""; $(NF-1)=""; $NF=""; print $0}' | sed 's/^[ ]*//;s/[ ]*$//')
        local path="/dev/$name"
        _debug "Parsed: Name='$name', Model='$model', Size='$size', Type='$type'"

        # Include all block devices (disk, rom for CDROMs/floppies)
        [ -b "$path" ] || { _debug "Skipping $path: Not a block device"; continue; }

        local is_nvme=0
        [[ "$name" == nvme* ]] && is_nvme=1

        local smart_ok=0 output=""
        if [ "$is_nvme" -eq 1 ] && command -v smartctl &>/dev/null; then
            _debug "Checking NVMe device $path with smartctl --device=nvme"
            local cmd=(smartctl --device=nvme -i "$path")
            [ -n "$SUDO_CMD" ] && cmd=("$SUDO_CMD" "${cmd[@]}")
            _debug "Executing: ${cmd[*]}"
            if output=$("${cmd[@]}" 2>&1); then
                echo "$output" | grep -qE "Model Number:|Serial Number:|Firmware Version:|NVMe Version:" && smart_ok=1
                _debug "smartctl --device=nvme -i: $( [ "$smart_ok" -eq 1 ] && echo "Success" || echo "Failed: $output" )"
            else
                _debug "smartctl --device=nvme -i failed: $output"
            fi
        fi

        if [ "$smart_ok" -eq 0 ] && [ "$is_nvme" -eq 1 ] && command -v nvme &>/dev/null; then
            _debug "Checking NVMe device $path with nvme-cli"
            local cmd=(nvme id-ctrl "$path")
            [ -n "$SUDO_CMD" ] && cmd=("$SUDO_CMD" "${cmd[@]}")
            _debug "Executing: ${cmd[*]}"
            if output=$("${cmd[@]}" 2>&1); then
                echo "$output" | grep -qE "vid|sn|mn|fr" && smart_ok=1
                _debug "nvme id-ctrl: $( [ "$smart_ok" -eq 1 ] && echo "Success" || echo "Failed: $output" )"
            else
                _debug "nvme id-ctrl failed: $output"
            fi
        fi

        if [ "$smart_ok" -eq 0 ] && [ "$is_nvme" -eq 0 ] && command -v smartctl &>/dev/null; then
            _debug "Checking $path with smartctl"
            local cmd=(smartctl -i "$path")
            [ -n "$SUDO_CMD" ] && cmd=("$SUDO_CMD" "${cmd[@]}")
            _debug "Executing: ${cmd[*]}"
            if output=$("${cmd[@]}" 2>&1); then
                echo "$output" | grep -qE "Model Number:|Serial Number:|Firmware Version:" && smart_ok=1
                _debug "smartctl -i: $( [ "$smart_ok" -eq 1 ] && echo "Success" || echo "Failed: $output" )"
            else
                _debug "smartctl -i failed: $output"
            fi
        fi

        local size_human=$(numfmt --to=iec-i --suffix=B --format="%.2f" "$size" 2>/dev/null || echo "N/A")
        DRIVE_PATHS+=("$path")
        DRIVE_NAMES+=("$path - ${model:-Unknown Model} ($size_human, Type: $type)")
        _debug "Added: $path"
    done <"$tmp_file"
    return 0
}

# --- Report Generation ---
_generate_report() {
    local device="$1"
    local report_file=$(mktemp -p "$REPORT_BASE_DIR" "smart_report_XXXXXX.txt") || { _error "Failed to create report file."; return 1; }
    TEMP_FILES+=("$report_file")
    _debug "Generating report for $device"

    local datetime=$(date "+%Y-%m-%d %H:%M:%S %Z")
    local model="N/A" serial="N/A" firmware="N/A" capacity="N/A"
    local health="" attrs="" info=""
    local health_status="Unknown" recommendation="Investigate detailed output below."
    local cmd_info=() cmd_health=() cmd_attrs=()

    local is_nvme=0
    [[ "$device" == /dev/nvme* ]] && is_nvme=1

    # Ask user to choose tool for NVMe devices
    local use_smartctl=1
    if [ "$is_nvme" -eq 1 ] && command -v nvme &>/dev/null && command -v smartctl &>/dev/null; then
        _info "NVMe device detected: $device"
        if _ask_yes_no "Use smartctl for health check (recommended)? (y/n)" y; then
            use_smartctl=1
        else
            use_smartctl=0
        fi
    fi

    if [ "$is_nvme" -eq 1 ] && [ "$use_smartctl" -eq 1 ] && command -v smartctl &>/dev/null; then
        _debug "Using smartctl --device=nvme for $device"
        cmd_info=(smartctl --device=nvme -i "$device")
        cmd_health=(smartctl --device=nvme -H "$device")
        cmd_attrs=(smartctl --device=nvme -A "$device")
        [ -n "$SUDO_CMD" ] && {
            cmd_info=("$SUDO_CMD" "${cmd_info[@]}")
            cmd_health=("$SUDO_CMD" "${cmd_health[@]}")
            cmd_attrs=("$SUDO_CMD" "${cmd_attrs[@]}")
        }
        _debug "Executing info: ${cmd_info[*]}"
        info=$("${cmd_info[@]}" 2>&1 || echo "ERROR: smartctl --device=nvme -i failed")
        _debug "Executing health: ${cmd_health[*]}"
        health=$("${cmd_health[@]}" 2>&1 || echo "ERROR: smartctl --device=nvme -H failed")
        _debug "Executing attrs: ${cmd_attrs[*]}"
        attrs=$("${cmd_attrs[@]}" 2>&1 || echo "ERROR: smartctl --device=nvme -A failed")
        model=$(echo "$info" | grep -E "^Model Number:|^Product:" | awk -F': ' '{print $2}' | head -n 1 | xargs || echo "N/A")
        serial=$(echo "$info" | grep "Serial Number:" | awk -F': ' '{print $2}' | xargs || echo "N/A")
        firmware=$(echo "$info" | grep "Firmware Version:" | awk -F': ' '{print $2}' | xargs || echo "N/A")
        capacity=$(echo "$info" | grep "Total NVM Capacity:" | awk -F': ' '{print $2}' | xargs || echo "N/A")
        if echo "$health" | grep -iq "PASSED"; then
            health_status="Healthy"
            recommendation="Drive is healthy. Regular checks recommended."
        elif echo "$health" | grep -iq "FAILED"; then
            health_status="Failed"
            recommendation="CRITICAL: Drive failure detected. Back up data immediately and replace the drive."
        else
            # Check attributes for warning signs (e.g., wear leveling, reallocated sectors)
            if echo "$attrs" | grep -iqE "Percentage Used:.*[1-9][0-9]*%|Media Wearout Indicator:.*[0-9]{1,2}%"; then
                health_status="Critical"
                recommendation="WARNING: Drive shows wear or errors. Back up data and monitor condition."
            fi
        fi
    elif [ "$is_nvme" -eq 1 ] && [ "$use_smartctl" -eq 0 ] && command -v nvme &>/dev/null; then
        _debug "Using nvme-cli for $device"
        cmd_info=(nvme id-ctrl "$device")
        cmd_health=(nvme smart-log "$device")
        cmd_attrs=("N/A" "(nvme-cli does not provide detailed attributes)")
        [ -n "$SUDO_CMD" ] && {
            cmd_info=("$SUDO_CMD" "${cmd_info[@]}")
            cmd_health=("$SUDO_CMD" "${cmd_health[@]}")
        }
        _debug "Executing info: ${cmd_info[*]}"
        info=$("${cmd_info[@]}" 2>&1 || echo "ERROR: nvme id-ctrl failed")
        _debug "Executing health: ${cmd_health[*]}"
        health=$("${cmd_health[@]}" 2>&1 || echo "ERROR: nvme smart-log failed")
        attrs="Note: Detailed attributes not available via nvme-cli. See smart-log output."
        model=$(echo "$info" | grep -i "mn" | awk -F': ' '{print $2}' | xargs || echo "N/A")
        serial=$(echo "$info" | grep -i "sn" | awk -F': ' '{print $2}' | xargs || echo "N/A")
        firmware=$(echo "$info" | grep -i "fr" | awk -F': ' '{print $2}' | xargs || echo "N/A")
        capacity=$(echo "$info" | grep -i "tnvmcap" | awk -F': ' '{print $2}' | xargs || echo "N/A")
        if echo "$health" | grep -q "Critical Warning:[[:space:]]*0"; then
            health_status="Healthy"
            recommendation="Drive is healthy. Regular checks recommended."
        elif echo "$health" | grep -q "Critical Warning:[[:space:]]*[1-9]"; then
            # Check for specific critical warnings
            if echo "$health" | grep -iqE "temperature.*threshold|available spare.*threshold"; then
                health_status="Critical"
                recommendation="WARNING: Drive reports critical conditions (e.g., temperature, spare capacity). Back up data and monitor condition."
            else
                health_status="Failed"
                recommendation="CRITICAL: Drive reports severe errors. Back up data immediately and replace the drive."
            fi
        else
            health_status="Critical"
            recommendation="WARNING: Smart log unavailable or faulty. Back up data and check condition."
        fi
    else
        _debug "Using smartctl for $device"
        cmd_info=(smartctl -i "$device")
        cmd_health=(smartctl -H "$device")
        cmd_attrs=(smartctl -A "$device")
        [ -n "$SUDO_CMD" ] && {
            cmd_info=("$SUDO_CMD" "${cmd_info[@]}")
            cmd_health=("$SUDO_CMD" "${cmd_health[@]}")
            cmd_attrs=("$SUDO_CMD" "${cmd_attrs[@]}")
        }
        _debug "Executing info: ${cmd_info[*]}"
        info=$("${cmd_info[@]}" 2>&1 || echo "ERROR: smartctl -i failed")
        _debug "Executing health: ${cmd_health[*]}"
        health=$("${cmd_health[@]}" 2>&1 || echo "ERROR: smartctl -H failed")
        _debug "Executing attrs: ${cmd_attrs[*]}"
        attrs=$("${cmd_attrs[@]}" 2>&1 || echo "ERROR: smartctl -A failed")
        model=$(echo "$info" | grep -E "^Device Model:|^Product:" | awk -F': ' '{print $2}' | head -n 1 | xargs || echo "N/A")
        serial=$(echo "$info" | grep "Serial Number:" | awk -F': ' '{print $2}' | xargs || echo "N/A")
        firmware=$(echo "$info" | grep "Firmware Version:" | awk -F': ' '{print $2}' | xargs || echo "N/A")
        capacity=$(echo "$info" | grep "User Capacity:" | awk -F': ' '{print $2}' | xargs || echo "N/A")
        if echo "$health" | grep -iq "PASSED"; then
            health_status="Healthy"
            recommendation="Drive is healthy. Regular checks recommended."
        elif echo "$health" | grep -iq "FAILED"; then
            health_status="Failed"
            recommendation="CRITICAL: Drive failure detected. Back up data immediately and replace the drive."
        else
            # Check attributes for warning signs (e.g., reallocated sectors)
            if echo "$attrs" | grep -iqE "Reallocated_Sector_Ct.*[1-9]|Wear_Leveling_Count.*[0-9]{1,2}%"; then
                health_status="Critical"
                recommendation="WARNING: Drive shows wear or errors. Back up data and monitor condition."
            fi
        fi
    fi

    cat <<EOF >"$report_file"
======================================================
$PROG_NAME - Drive Health Report
======================================================
Generated: $datetime
Debug Mode: $( [ "$_DEBUG" -eq 1 ] && echo "YES" || echo "NO" )
Author: ZeroDot1 <https://github.com/ZeroDot1
Donate: https://www.amazon.de/hz/wishlist/ls/2DDEDPJU2996I/

--- Device Info ---
Path: $device
Model: $model
Serial: $serial
Firmware: $firmware
Capacity: $capacity

--- Health Assessment ---
Health Status: $health_status
Recommendation: $recommendation

--- System Info ---
$(_get_system_info)

--- Commands Used ---
Identification: $(printf "%q " "${cmd_info[@]}")
Health Check: $(printf "%q " "${cmd_health[@]}")
Attributes: $(printf "%q " "${cmd_attrs[@]}")

--- SMART/NVMe Attributes ---
$attrs

--- SMART/NVMe Health ---
$health

--- Identification Info ---
$info

======================================================
End of Report
======================================================
EOF

    _info "Report generated: $report_file"
    echo "$report_file"
}

# --- Report Display ---
_display_report() {
    local file="$1"
    [ -f "$file" ] || { _error "Report file $file not found."; return 1; }
    _info "Displaying report..."
    if command -v leafpad &>/dev/null; then
        leafpad "$file" &
        _pause "Review the report in Leafpad"
    elif command -v less &>/dev/null; then
        less -R "$file"
    else
        cat "$file"
        _pause "End of report (install leafpad or less for better viewing)"
    fi
}

# --- Main ---
main() {
    [ "$#" -ge 1 ] && [ "$1" = "--debug" ] && _DEBUG=1 && _info "Debug mode enabled"
    clear
    echo "======================================================" >&2
    echo "$PROG_NAME - Drive Health Check Utility" >&2
    echo "A minimalist tool for SMART/NVMe reports on Arch Linux" >&2
    echo "Author: ZeroDot1 <https://github.com/ZeroDot1>" >&2
    echo "Donate: https://www.amazon.de/hz/wishlist/ls/2DDEDPJU2996I/" >&2
    echo "======================================================" >&2

    _set_sudo
    _check_deps

    while true; do
        clear
        _info "Scanning drives..."
        _discover_drives || { _error "Drive discovery failed."; _pause "Exiting"; exit 1; }

        if [ ${#DRIVE_PATHS[@]} -eq 0 ]; then
            _pause "No drives found. Check connections, permissions, or install nvme-cli/smartmontools."
            _ask_yes_no "Scan again?" || break
            continue
        fi

        echo -e "\n--- Available Drives ---" >&2
        PS3=$'\nSelect a drive (or "q" to quit): '
        select name in "${DRIVE_NAMES[@]}" "Quit"; do
            [[ "$name" == "Quit" ]] && break 2
            local idx=$((REPLY - 1))
            if [ "$idx" -ge 0 ] && [ "$idx" -lt "${#DRIVE_PATHS[@]}" ]; then
                local device="${DRIVE_PATHS[$idx]}"
                _info "Selected: $device"
                local report=$(_generate_report "$device") || { _error "Report generation failed."; _pause "Returning to menu"; break; }
                _display_report "$report"
                _ask_yes_no "Check another drive?" || break 2
            else
                _error "Invalid selection. Try again."
            fi
            break
        done
    done

    clear
    _info "Exiting $PROG_NAME."
}

main "$@"