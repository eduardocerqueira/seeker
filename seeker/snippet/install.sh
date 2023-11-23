#date: 2023-11-23T16:41:27Z
#url: https://api.github.com/gists/f161ef1ee3efdff07f5ff3d7b036c664
#owner: https://api.github.com/users/fcoury

#!/bin/bash

set -euo pipefail

systemctl start NetworkManager
pacman -S neofetch
neofetch

pacman -Syu
pacman -S sudo

useradd -m -g users -G wheel,storage,power,audio fcoury
passwd fcoury

cp /etc/sudoers /etc/sudoers.bak
sed -i '/^#\s*%wheel ALL=(ALL:ALL) NOPASSWD: ALL/s/^#//' /etc/sudoers

run_as_fcoury() {
    su - fcoury -c "$1"
}

run_sudo() {
    sudo $1
}

run_sudo "pacman -S xdg-user-dirs"
run_as_fcoury "xdg-user-dirs-update"

run_sudo "pacman -S git"
run_as_fcoury "cd /home/fcoury && git clone https://aur.archlinux.org/yay.git && cd yay && makepkg -si"

run_sudo "pacman -S spice-vdagent"
run_sudo "pacman -S pulseaudio alsa-utils alsa-plugins pavucontrol"
run_sudo "pacman -S openssh iw wpa_supplicant"
run_sudo "systemctl enable sshd"
run_sudo "systemctl enable dhcpcd"
run_sudo "systemctl enable NetworkManager"

run_sudo "pacman -S bluez bluez-utils blueman"
run_sudo "systemctl enable bluetooth"

run_sudo "sed -i '/^#Color/s/^#//' /etc/pacman.conf"
run_sudo "sed -i '/^Color/a ILoveCandy' /etc/pacman.conf"

run_sudo "systemctl enable fstrim.timer"
run_sudo "pacman -S xorg-server xorg-apps xorg-xinit xclip"
run_sudo "pacman -S i3"

echo "exec i3" > ~/.xinitrc

run_sudo "pacman -S picom"
run_sudo "pacman -S noto-fonts ttf-cascadia-code-nerd ttf-firacode-nerd ttf-iosevka-nerd ttf-jetbrains-mono-nerd ttf-cascadia-code-nerd ttf-nerd-fonts-symbols ttf-victor-mono-nerd"

run_sudo "pacman -S zsh alacritty ranger dmenu rofi polybar feh ueberzug wget fd ripgrep htop"

echo "Done."
