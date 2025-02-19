#date: 2025-02-19T16:50:39Z
#url: https://api.github.com/gists/62655014cd6d1154abb0bf3a77265ce8
#owner: https://api.github.com/users/ruancomelli

#!/bin/bash

# A script to upgrade all packages and check if a reboot is required
# Save this file as `up.sh` or add the function directly to `~/.bashrc` or `~/.zshrc`
# To make this script executable and accessible globally, move it to `/usr/local/bin/up` and run `chmod +x /usr/local/bin/up`

up() {
    echo -e "\033[1m→ Upgrading packages\033[0m"
    if command -v apt >/dev/null; then
        sudo apt update --fix-missing
        sudo apt install --fix-broken
        sudo apt full-upgrade -y
        sudo apt autoremove -y --purge
        sudo apt autoclean -y
    fi

    echo -e "\033[1m→ Upgrading cargo packages\033[0m"
    if command -v cargo >/dev/null; then
        cargo install-update -a
    fi

    echo -e "\033[1m→ Upgrading mise\033[0m"
    if command -v mise >/dev/null; then
        mise self-update -y
        mise up -y --bump
    fi

    echo -e "\033[1m→ Upgrading uv\033[0m"
    if command -v uv >/dev/null; then
        uv self update
        uv tool upgrade --all
    fi

    echo -e "\033[1m→ Upgrading rust\033[0m"
    if command -v rustup >/dev/null; then
        rustup update
    fi

    echo -e "\033[1m→ Upgrading Flatpak packages\033[0m"
    if command -v flatpak >/dev/null; then
        flatpak update -y
    fi

    echo -e "\033[1m→ Upgrading Snap packages\033[0m"
    if command -v snap >/dev/null; then
        sudo snap refresh
    fi

    echo -e "\033[1m→ Cleaning up Docker\033[0m"
    if command -v docker >/dev/null; then
        docker system prune -af
    fi

    echo -e "\033[1m→ Checking if a reboot is required\033[0m"
    if [ -f /var/run/reboot-required ]; then
        echo -e "\033[1m→ Reboot required\033[0m"
    else
        echo -e "\033[1m→ No reboot required\033[0m"
    fi
}

# If added to `~/.bashrc` or `~/.zshrc`, source the file to apply changes:
# source ~/.bashrc   # For Bash
# source ~/.zshrc    # For Zsh

# To schedule this function to run daily at 6 AM, add the following line to your crontab:
# crontab -e
# 0 6 * * * bash -c '. ~/.bashrc && up'  # Adjust to `~/.zshrc` if using Zsh

# To run it manually, simply type:
# up