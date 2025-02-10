#date: 2025-02-10T16:55:38Z
#url: https://api.github.com/gists/da1b4302afac8e0edd8d2af9100c2967
#owner: https://api.github.com/users/EvilSupahFly

#!/usr/bin/env bash

# Function to check if color output is supported
supports_color() {
    if ! [ -t 1 ]; then
        return 1  # No color support (output is being piped)
    fi

    if command -v tput &> /dev/null && [ "$(tput colors)" -ge 8 ]; then
        return 0  # Supports color
    fi

    return 1  # No color support
}

# Define colors only if supported
if supports_color; then
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    PURPLE=$(tput setaf 5)
    WHITE=$(tput setaf 7)
    RESET=$(tput sgr0)
else
    RED=""
    GREEN=""
    YELLOW=""
    PURPLE=""
    WHITE=""
    RESET=""
fi

# Enable case-insensitive globbing for better directory matching
shopt -s nocaseglob

# Function to determine the package manager
detect_package_manager() {
    if command -v apt &> /dev/null; then
        echo "apt"
    elif command -v dnf &> /dev/null; then
        echo "dnf"
    elif command -v pacman &> /dev/null; then
        echo "pacman"
    elif command -v zypper &> /dev/null; then
        echo "zypper"
    elif command -v yum &> /dev/null; then
        echo "yum"
    elif command -v brew &> /dev/null; then
        echo "brew"
    else
        echo "unknown"
    fi
}

# Function to install missing dependencies
install_package() {
    local package="$1"
    local manager
    manager=$(detect_package_manager)

    if [ "$manager" = "unknown" ]; then
        echo -e "${RED}Could not detect a package manager. Please install $package manually.${RESET}"
        exit 1
    fi

    echo -e "${YELLOW}$package is missing. Would you like to install it now? (y/n)${RESET}"
    read -rp "Enter choice: " choice

    if [[ "$choice" =~ ^[Yy]$ ]]; then
        case "$manager" in
            apt) sudo apt update && sudo apt install -y "$package" ;;
            dnf) sudo dnf install -y "$package" ;;
            pacman) sudo pacman -Sy --noconfirm "$package" ;;
            zypper) sudo zypper install -y "$package" ;;
            yum) sudo yum install -y "$package" ;;
            brew) brew install "$package" ;;
        esac

        if command -v "$package" &> /dev/null; then
            echo -e "${GREEN}$package installed successfully.${RESET}"
        else
            echo -e "${RED}Failed to install $package. Please install it manually.${RESET}"
            exit 1
        fi
    else
        echo -e "${RED}$package is required to run this script. Exiting.${RESET}"
        exit 1
    fi
}

# Check for required dependencies
for cmd in wine curl jq; do
    if ! command -v "$cmd" &> /dev/null; then
        install_package "$cmd"
    fi
done

# Ensure script is not run as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Do not run this script as root. Exiting.${RESET}"
    exit 1
fi

# Function to list and select a directory
select_directory() {
    local search_dir="$1"
    local dirs=("$search_dir"/*/)

    if [ "${#dirs[@]}" -eq 0 ]; then
        echo -e "${RED}No directories found in $search_dir.${RESET}"
        return 1
    fi

    echo -e "${WHITE}Select a directory:${RESET}"
    for i in "${!dirs[@]}"; do
        echo -e "${GREEN}$((i + 1)).${RESET} ${YELLOW}${dirs[$i]##*/}${RESET}"
    done

    local choice
    while true; do
        read -rp "Enter the number of the directory: " choice
        if [[ "$choice" -ge 1 && "$choice" -le "${#dirs[@]}" ]]; then
            echo "${dirs[$((choice - 1))]}"
            return 0
        else
            echo -e "${RED}Invalid selection. Please try again.${RESET}"
        fi
    done
}

# Function to confirm settings before proceeding
final_confirmation() {
    while true; do
        echo -e "${PURPLE}-------------------------------------${RESET}"
        echo -e "${WHITE} Final Confirmation${RESET}"
        echo -e "${PURPLE}-------------------------------------${RESET}"
        echo -e "${GREEN}1.${RESET} WINE Prefix: ${YELLOW}$WINEPREFIX${RESET}"
        echo -e "${GREEN}2.${RESET} Installer Path: ${YELLOW}${INSTALLER_PATH:-'Not Set'}${RESET}"
        echo -e "${GREEN}3.${RESET} Selected EXE: ${YELLOW}${GAME_EXE_PATH:-'Not Set'}${RESET}"
        echo -e "${GREEN}4.${RESET} Proceed with installation"
        echo -e "${GREEN}5.${RESET} Cancel Installation"
        echo -e "${PURPLE}-------------------------------------${RESET}"

        read -rp "Select an option to modify or confirm (1-5): " choice
        case "$choice" in
            1) read -rp "Enter new WINE prefix: " WINEPREFIX ;;
            2) read -rp "Enter new installer path: " INSTALLER_PATH ;;
            3) GAME_EXE_PATH=$(select_directory "$HOME/Games") ;;
            4) echo -e "${GREEN}Proceeding with installation...${RESET}"; break ;;
            5) echo -e "${RED}Installation canceled.${RESET}"; exit 1 ;;
            *) echo -e "${RED}Invalid choice, please try again.${RESET}" ;;
        esac
    done
}

# Start script execution
echo -e "${WHITE}Welcome to the WINE Game Installer!${RESET}"
echo -e "${WHITE}Scanning for game directories...${RESET}"

GAMES_DIR="$HOME/Games"
SELECTED_DIR=$(select_directory "$GAMES_DIR")
if [ -z "$SELECTED_DIR" ]; then
    echo -e "${RED}No valid game directory selected. Exiting.${RESET}"
    exit 1
fi

WINEPREFIX="$HOME/.wine"
INSTALLER_PATH="$SELECTED_DIR/installer.exe"
GAME_EXE_PATH="$SELECTED_DIR/game.exe"

final_confirmation

# Run the installer
if [ -f "$INSTALLER_PATH" ]; then
    echo -e "${WHITE}Running installer: ${YELLOW}$INSTALLER_PATH${RESET}"
    wine "$INSTALLER_PATH"
fi

echo -e "${GREEN}Installation complete!${RESET}"
