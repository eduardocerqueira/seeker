#date: 2025-09-24T17:09:27Z
#url: https://api.github.com/gists/3b693e1e012ccdee4410b20ea848479b
#owner: https://api.github.com/users/gustavomdsantos

#!/bin/bash

# ==============================================================================
# Script: wifi-manager-setup.sh
# Description: Installer and uninstaller for the Wi-Fi management daemon.
# Author: Gustavo Moraes <gustavomdsantos@pm.me>
# ==============================================================================

# --- Variables ----------------------------------------------------------------
DAEMON_SCRIPT="wifi-manager-daemon.sh"
SERVICE_NAME="wifi-manager.service"
SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}"
DAEMON_PATH="/usr/local/bin/${DAEMON_SCRIPT}"

# --- Utility Functions --------------------------------------------------------

# Displays a multi-line message using a single echo command.
display_message() {
    local message="$1"
    echo -e "$message"
}

# Checks for required commands and provides helpful error messages.
check_dependencies() {
    local missing_deps=()
    local required_deps=(
        "systemctl" 
        "ip"
        "lsusb"
        "lspci"
        "awk"
        "grep"
    )

    for dep in "${required_deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done

    if [[ ${#missing_deps[@]} -ne 0 ]]; then
        local message
        message=$(cat <<EOF
ERROR: The following required dependencies are not installed:
$(printf "* %s\n" "${missing_deps[@]}")

Please install them and try again.
EOF
)
        display_message "$message"
        exit 1
    fi
}

# Verifies if the script is running as root.
check_root() {
    if [[ $EUID -ne 0 ]]; then
        display_message "ERROR: This script must be run as root. Please use 'sudo'."
        exit 1
    fi
}

# --- Service Management Functions ---------------------------------------------

# Creates and writes the systemd service file.
create_service_file() {
    local service_content
    service_content=$(cat <<EOF
[Unit]
Description=Wi-Fi Manager Daemon
After=network-online.target

[Service]
ExecStart=${DAEMON_PATH}
Restart=on-failure
RestartSec=5s
User=root

[Install]
WantedBy=multi-user.target
EOF
)
    echo "${service_content}" | tee "${SERVICE_FILE_PATH}" &>/dev/null
    if [[ $? -ne 0 ]]; then
        display_message "ERROR: Failed to create systemd service file at ${SERVICE_FILE_PATH}."
        exit 1
    fi
}

# Installs the daemon and the systemd service.
install_service() {
    display_message "--- Starting installation of the Wi-Fi Manager Daemon ---"

    if [[ -f "${DAEMON_PATH}" ]]; then
        display_message "INFO: Daemon script already exists. Overwriting..."
    fi

    local daemon_script
    daemon_script=$(cat << 'EOF_DAEMON'
#!/bin/bash

# ==============================================================================
# Script: wifi-manager-daemon.sh
# Description: Daemon to manage Wi-Fi interfaces automatically.
# Author: Gustavo Moraes <gustavomdsantos@pm.me>
# ==============================================================================

# --- Utility Functions --------------------------------------------------------

# Displays a multi-line message. This function is now local to the daemon.
display_message() {
    local message="$1"
    echo -e "$message"
}

# Finds the internal PCI Wi-Fi interface.
find_internal_wifi() {
    local pci_interfaces
    pci_interfaces=$(lspci | grep -i "wireless\|wi-fi\|network" | awk '{print $1}')
    
    for pci_id in ${pci_interfaces}; do
        local iface_name
        iface_name=$(ls -l /sys/class/net/ | grep "${pci_id}" | awk '{print $9}')
        if [[ -n "${iface_name}" ]] && [[ ! "${iface_name}" =~ ^wlx ]]; then
            echo "${iface_name}"
            return
        fi
    done
}

# Finds all USB Wi-Fi interfaces.
find_usb_wifi() {
    local usb_interfaces=()
    local all_interfaces
    all_interfaces=$(ip -o link show | awk -F': ' '/wlx/ {print $2}')

    for iface in ${all_interfaces}; do
        usb_interfaces+=("${iface}")
    done
    echo "${usb_interfaces[@]}"
}

# Checks if a Wi-Fi interface exists and is enabled.
is_interface_enabled() {
    local interface="$1"
    ip link show "$interface" &> /dev/null && ip link show "$interface" | grep -q "state UP"
}

# Brings a Wi-Fi interface up.
enable_interface() {
    local interface="$1"
    if ! is_interface_enabled "${interface}"; then
        if ! ip link set "$interface" up; then
            display_message "ERROR: Failed to enable interface '$interface'."
        fi
    fi
}

# Brings a Wi-Fi interface down.
disable_interface() {
    local interface="$1"
    if is_interface_enabled "${interface}"; then
        if ! ip link set "$interface" down; then
            display_message "ERROR: Failed to disable interface '$interface'."
        fi
    fi
}

# --- Main Logic ---------------------------------------------------------------
display_message "INFO: Wi-Fi manager daemon started."

# Declare variables for state management.
internal_interface=""
usb_interfaces=()
usb_connected=false
current_state="unknown"

while true; do
    internal_interface=$(find_internal_wifi)
    usb_interfaces=($(find_usb_wifi))
    usb_connected=false

    if [[ -z "${internal_interface}" ]]; then
        display_message "ERROR: Could not find the internal Wi-Fi interface. The daemon will exit."
        exit 1
    fi

    # Check for the presence of any active USB Wi-Fi interface
    for iface in "${usb_interfaces[@]}"; do
        if ip link show "${iface}" &> /dev/null; then
            usb_connected=true
            break
        fi
    done

    if ${usb_connected}; then
        if [[ "${current_state}" != "usb" ]]; then
            display_message "INFO: USB Wi-Fi adapter detected. Disabling internal interface '${internal_interface}'."
            disable_interface "${internal_interface}"
            current_state="usb"
        fi
    else
        if [[ "${current_state}" != "internal" ]]; then
            display_message "INFO: No USB Wi-Fi adapter found. Enabling internal interface '${internal_interface}'."
            enable_interface "${internal_interface}"
            current_state="internal"
        fi
    fi

    # Check for changes every 2 seconds.
    sleep 2
done
EOF_DAEMON
)
    echo "${daemon_script}" | tee "${DAEMON_PATH}" &>/dev/null
    if [[ $? -ne 0 ]]; then
        display_message "ERROR: Failed to write daemon script to ${DAEMON_PATH}."
        exit 1
    fi

    chmod +x "${DAEMON_PATH}"

    create_service_file

    display_message "INFO: Reloading systemd daemon..."
    systemctl daemon-reload

    display_message "INFO: Enabling the Wi-Fi Manager service..."
    systemctl enable "${SERVICE_NAME}"

    display_message "INFO: Wi-Fi Manager daemon and service installed successfully."
    display_message "Use 'sudo systemctl status ${SERVICE_NAME}' to check its status."
    display_message "Installation complete."
}

# Uninstalls the daemon and the systemd service.
uninstall_service() {
    display_message "--- Starting uninstallation of the Wi-Fi Manager Daemon ---"

    if systemctl is-active --quiet "${SERVICE_NAME}"; then
        display_message "INFO: Stopping the service..."
        systemctl stop "${SERVICE_NAME}"
    fi

    if systemctl is-enabled --quiet "${SERVICE_NAME}"; then
        display_message "INFO: Disabling the service..."
        systemctl disable "${SERVICE_NAME}"
    fi
    
    if [[ -f "${SERVICE_FILE_PATH}" ]]; then
        display_message "INFO: Removing systemd service file..."
        rm -f "${SERVICE_FILE_PATH}"
    fi

    if [[ -f "${DAEMON_PATH}" ]]; then
        display_message "INFO: Removing daemon script..."
        rm -f "${DAEMON_PATH}"
    fi

    display_message "INFO: Reloading systemd daemon..."
    systemctl daemon-reload
    
    display_message "INFO: All files have been removed."
    display_message "Uninstallation complete."
}

# Starts the systemd service.
start_service() {
    check_root
    if [[ ! -f "${SERVICE_FILE_PATH}" ]]; then
        display_message "ERROR: Service file not found. Please run 'sudo ./wifi-manager-setup.sh install' first."
        exit 1
    fi
    display_message "INFO: Starting the Wi-Fi Manager service..."
    systemctl enable "${SERVICE_NAME}"
    systemctl start "${SERVICE_NAME}"
    display_message "INFO: Service started. Use 'sudo systemctl status ${SERVICE_NAME}' to check its status."
}

# Stops the systemd service.
stop_service() {
    check_root
    if [[ ! -f "${SERVICE_FILE_PATH}" ]]; then
        display_message "ERROR: Service file not found. It seems the daemon is not installed."
        exit 1
    fi
    display_message "INFO: Stopping the Wi-Fi Manager service..."
    systemctl stop "${SERVICE_NAME}"
    display_message "INFO: Service stopped."
}

# --- Main Logic (Command Parsing) ---------------------------------------------
main() {
    if [[ -z "$1" ]]; then
        local usage_message
        usage_message=$(cat <<EOF
Usage: $0 {install|uninstall|start|stop}

Commands:
  install      Installs the daemon script and systemd service.
  uninstall    Uninstalls the daemon script and systemd service.
  start        Starts the daemon service (must be installed first).
  stop         Stops the daemon service.
EOF
)
        display_message "$usage_message"
        exit 1
    fi

    case "$1" in
        install)
            check_root
            check_dependencies
            install_service
            ;;
        uninstall)
            check_root
            uninstall_service
            ;;
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        *)
            display_message "ERROR: Invalid command '$1'. Use 'install', 'uninstall', 'start' or 'stop'."
            exit 1
            ;;
    esac
}

main "$@"

