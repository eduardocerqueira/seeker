#date: 2025-05-14T17:11:13Z
#url: https://api.github.com/gists/1213d1c95fceb0dd4f01e95727e14977
#owner: https://api.github.com/users/AbdulrahmanSolioman

#!/bin/bash

Full Arch Linux + Hyprland + Catppuccin + ADHD-Friendly Setup

WARNING: This will wipe your disk. Only run this on a fresh system.

Boot from Arch ISO, connect to internet, then run this script.

set -e

---------------------------

0. Variables

---------------------------

DISK= "**********"=arch USER=archuser PASSWORD=archpass

---------------------------

1. Partition and Format

---------------------------

echo "[1/12] Partitioning $DISK..." sgdisk -Z $DISK sgdisk -n1:0:+512M -t1:ef00 -c1:EFI $DISK sgdisk -n2:0:0 -t2:8300 -c2:ROOT $DISK mkfs.fat -F32 ${DISK}1 mkfs.ext4 ${DISK}2

mount ${DISK}2 /mnt mkdir /mnt/boot mount ${DISK}1 /mnt/boot

---------------------------

2. Install Base System

---------------------------

echo "[2/12] Installing base system..." pacstrap /mnt base linux linux-firmware vim sudo networkmanager grub efibootmgr git

---------------------------

3. Generate Fstab

---------------------------

genfstab -U /mnt >> /mnt/etc/fstab

---------------------------

4. Configure System

---------------------------

echo "[3/12] Configuring system..." arch-chroot /mnt bash -c "echo $HOST > /etc/hostname" arch-chroot /mnt bash -c "ln -sf /usr/share/zoneinfo/UTC /etc/localtime && hwclock --systohc" arch-chroot /mnt bash -c "sed -i 's/^#en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen" echo "LANG=en_US.UTF-8" > /mnt/etc/locale.conf

---------------------------

5. Create User

---------------------------

arch-chroot /mnt bash -c "useradd -m -G wheel -s /bin/bash $USER" echo "$USER: "**********":$PASSWORD" | arch-chroot /mnt chpasswd arch-chroot /mnt bash -c "echo '%wheel ALL=(ALL:ALL) ALL' >> /etc/sudoers"

---------------------------

6. Enable Services

---------------------------

arch-chroot /mnt systemctl enable NetworkManager

---------------------------

7. Bootloader

---------------------------

arch-chroot /mnt grub-install --target=x86_64-efi --efi-directory=/boot --bootloader-id=GRUB arch-chroot /mnt grub-mkconfig -o /boot/grub/grub.cfg

---------------------------

8. Yay + Hyprland Setup

---------------------------

echo "[8/12] Installing yay and Hyprland packages..." arch-chroot /mnt bash -c "sudo -u $USER bash -c ' cd ~ && git clone https://aur.archlinux.org/yay.git && cd yay && makepkg -si --noconfirm && yay -Syu --noconfirm && yay -S --noconfirm hyprland kitty waybar wofi dunst rofi cliphist \ swww nwg-look lxappearance papirus-icon-theme \ zoxide btop spicetify zathura gamemode mangohud \ qt5ct qt6ct xdg-desktop-portal-hyprland \ swaylock swayidle grim slurp swappy '"

---------------------------

9. Catppuccin Theming

---------------------------

echo "[9/12] Setting up Catppuccin..." arch-chroot /mnt bash -c "sudo -u $USER bash -c ' mkdir -p ~/.config && cd ~/.config && git clone https://github.com/catppuccin/hyprland.git && git clone https://github.com/catppuccin/waybar.git && git clone https://github.com/catppuccin/dunst.git && git clone https://github.com/catppuccin/wofi.git && git clone https://github.com/catppuccin/kitty.git '"

---------------------------

10. Config Files

---------------------------

echo "[10/12] Creating user configs..." arch-chroot /mnt bash -c "sudo -u $USER bash -c ' mkdir -p ~/.config/hypr ~/.config/waybar ~/.config/dunst ~/.config/wofi ~/.config/kitty ~/notes ~/Pictures/wallpapers cp ~/catppuccin/hyprland/themes/mocha.conf ~/.config/hypr/mocha.conf cat > /.config/hypr/mocha.conf exec-once = waybar & exec-once = dunst & exec-once = swww init && swww img ~/Pictures/wallpapers/catppuccin.png exec-once = lxappearance & exec-once = cliphist store & exec-once = swayidle -w & exec-once = wl-paste --watch cliphist store & env = XCURSOR_THEME, Catppuccin-Mocha bind = SUPER, RETURN, exec, kitty bind = SUPER, W, killactive, bind = SUPER, E, exec, dolphin bind = SUPER, D, exec, wofi --show drun bind = SUPER, Q, exit, bind = SUPER, F, togglefloating, bind = SUPER, T, exec, kitty --class scratchpad bind = SUPER_SHIFT, T, exec, notify-send "Quick Note" && kitty --class scratchpad -e nvim ~/notes/quick.md windowrulev2 = float, class:^(scratchpad)$ windowrulev2 = center, class:^(scratchpad)$ windowrulev2 = opacity 0.95, class:^(scratchpad)$ workspace = 1, monitor:DP-1 workspace = 2, monitor:HDMI-A-1 EOF cp ~/catppuccin/waybar/themes/mocha.css ~/.config/waybar/style.css cp ~/catppuccin/dunst/themes/mocha.conf ~/.config/dunst/dunstrc cp ~/catppuccin/wofi/themes/mocha.css ~/.config/wofi/style.css cp ~/catppuccin/kitty/themes/mocha.conf ~/.config/kitty/kitty.conf '"

---------------------------

11. GTK Theme

---------------------------

arch-chroot /mnt bash -c "sudo -u $USER bash -c ' git clone https://github.com/catppuccin/gtk.git ~/.config/catppuccin-gtk && cd ~/.config/catppuccin-gtk && ./install.sh mocha '"

---------------------------

12. Done

---------------------------

echo "\n✅ All done. Reboot into your system and log into Hyprland as $USER." echo "📝 Quick notes: SUPER+SHIFT+T | Scratchpad: SUPER+T" echo "🎨 Set icons/theme with lxappearance" echo "🎮 Gaming: gamemoderun mangohud <game>" echo "⚠️ Reminder: Change wallpaper at ~/Pictures/wallpapers/catppuccin.png"

ictures/wallpapers/catppuccin.png"

