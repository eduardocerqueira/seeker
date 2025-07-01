#date: 2025-07-01T16:48:41Z
#url: https://api.github.com/gists/deff5f1b0986ad00bca1c8bb8bd5c90f
#owner: https://api.github.com/users/makeevrserg

is_installed() {
    local package="$1"
    if pacman -Qs "$package" > /dev/null; then
        echo "Package $package is installed"
        return 0
    else
        echo "Package $package is not installed"
        return 1
    fi
}

install_package() {
    local package="$1"
    sudo pacman -Syy --noconfirm "$package"
}

try_install_package() {
    local package="$1"
    if ! is_installed $package; then
        install_package $package
    fi
}

install_aur_package() {
    local package="$1"
    echo "Installing $package from AUR..."
    yay -S --noconfirm "$package"
}


try_install_aur_package() {
    local package="$1"
    if ! is_installed "$package"; then
        install_aur_package "$package"
    fi
}


if ! is_installed "docker"; then
    install_package "docker"
    sudo systemctl enable docker.service
    sudo systemctl start docker.service
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
fi
try_install_package "docker-compose"



try_install_package "gtk2"
try_install_package "libxcrypt-compat"
try_install_package "telegram-desktop"
try_install_package "vscode"
try_install_package "cpupower"
try_install_package "thermald"
try_install_package "wget"
try_install_package "btop"
try_install_aur_package "termius"
try_install_aur_package "slack-desktop"

# https://github.com/amnezia-vpn/amnezia-client/issues/792
fix_dns_vnp() {
    echo "[main]
    dns=none" | sudo tee /etc/NetworkManager/NetworkManager.conf > /dev/null

    echo "nameserver 1.1.1.1
    nameserver 1.0.0.1" | sudo tee /etc/resolv.conf > /dev/null

    sudo systemctl enable systemd-resolved.service && sudo systemctl start systemd-resolved.service
    sudo ln -sf ../run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
}

fix_dns_vnp

if ! pgrep -x "AmneziaVPN" > /dev/null; then
    echo "To continue, launch VPN"
    exit 1
fi

# https://github.com/amnezia-vpn/amnezia-client/releases/latest
# https://www.jetbrains.com/toolbox-app/download/download-thanks.html?platform=linux

# Remove intel_pstate
# sudo nano /etc/kernel/cmdline
# Add at the end intel_pstate=disable
# sudo find / -name "arch.conf"
# /usr/share/systemd/bootctl/arch.conf
# Add at the end intel_pstate=disable
# sudo bootctl update
# sudo reinstall-kernels
# sudo reboot now
# cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_driver
