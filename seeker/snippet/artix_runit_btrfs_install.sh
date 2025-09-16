#date: 2025-09-16T17:02:56Z
#url: https://api.github.com/gists/500a9517ce0cea0a44a89f8432f163bc
#owner: https://api.github.com/users/voloferenc

#!/bin/bash

# 0 - Log in as root (password is artix)
sudo su

# 1 - SSH
# This isn't necessary but if you ssh into the computer all the other steps are copy and paste
# Get network access
connmanctl

"""
# First, list all technologies: 
connmanctl> technologies
# Then, enable technology if it is not enabled
connmanctl> enable <technology>
# Then, to scan for services: 
connmanctl> scan <technology>
# You can then list all available services: 
connmanctl> services
# Finally, to connect to a service (use tab completion): 
connmanctl> connect <service>
"""

# Start the ssh daemon
ln -s /etc/runit/sv/sshd/ /etc/runit/runsvdir/default/
sv start sshd

# 2 - Partitioning:
cfdisk /dev/sda
# sda1 = /boot, sda2 = root
# /dev/sda1    512M          EFI System
# /dev/sda2    (the rest)    Linux Filesystem  

# 3 - Formatting the partitions:
# the first one is our ESP partition, so for now we just need to format it
mkfs.vfat -F32 -n "EFI" /dev/sda1
mkfs.btrfs -L ROOT /dev/sda2

# 4 - Create and Mount Subvolumes
# Create subvolumes for root, home, the package cache, snapshots and the entire Btrfs file system
mount /dev/sda2 /mnt
btrfs sub create /mnt/@artix
btrfs sub create /mnt/@home
btrfs sub create /mnt/@swap
btrfs sub create /mnt/@btrfs
btrfs sub create /mnt/@srv
btrfs sub create /mnt/@log
btrfs sub create /mnt/@tmp
btrfs sub create /mnt/@abs
btrfs sub create /mnt/@pkg
btrfs sub create /mnt/@snapshots
umount /mnt
# Mount the subvolumes
# Change space_cache to space_cache=v2 if there are errors
mount -o noatime,nodiratime,compress=zstd,commit=120,space_cache,ssd,discard=async,autodefrag,subvol=@artix /dev/sda2 /mnt
mkdir -p /mnt/{boot,home,swap,btrfs,srv,tmp,.snapshots,var/{log,tmp,abs,cache/pacman/pkg}}
mount -o noatime,nodiratime,compress=zstd,commit=120,space_cache,ssd,discard=async,autodefrag,subvol=@home /dev/sda2 /mnt/home
mount -o noatime,nodiratime,compress=zstd,commit=120,space_cache,ssd,discard=async,autodefrag,subvol=@srv /dev/sda2 /mnt/srv
mount -o noatime,nodiratime,compress=zstd,commit=120,space_cache,ssd,discard=async,autodefrag,subvol=@log /dev/sda2 /mnt/var/log
mount -o noatime,nodiratime,compress=zstd,commit=120,space_cache,ssd,discard=async,autodefrag,subvol=@tmp /dev/sda2 /mnt/var/tmp
mount -o noatime,nodiratime,compress=zstd,commit=120,space_cache,ssd,discard=async,autodefrag,subvol=@abs /dev/sda2 /mnt/var/abs
mount -o noatime,nodiratime,compress=zstd,commit=120,space_cache,ssd,discard=async,autodefrag,subvol=@pkg /dev/sda2 /mnt/var/cache/pacman/pkg
mount -o noatime,nodiratime,compress=zstd,commit=120,space_cache,ssd,discard=async,autodefrag,subvol=@snapshots /dev/sda2 /mnt/.snapshots

mount -o noatime,nodiratime,compress=zstd,commit=120,space_cache,ssd,discard=async,autodefrag,subvolid=5 /dev/sda2 /mnt/btrfs
mount -o compress=no,space_cache,ssd,discard=async,subvol=@swap /dev/sda2 /mnt/swap
mount -t tmpfs -o rw,nosuid,nodev,exec,auto,nouser,async,noatime,mode=1777,size=8G /dev/sda2 /mnt/tmp

# Create Swapfile
SWAPFILE=/mnt/swap/swapfile
truncate -s 0 $SWAPFILE
chattr +C $SWAPFILE
btrfs property set $SWAPFILE compression none
fallocate -l 16G $SWAPFILE
chmod 600 $SWAPFILE
mkswap $SWAPFILE
swapon $SWAPFILE

# Mount the EFI partition
mount /dev/sda1 /mnt/boot

# 5 Base System and /etc/fstab
# (this is the time where you change the mirrorlist, if that's your thing)
# (Optional) Enable parallel downloads for basestrap
sed -i 's/^#ParallelDownloads.*/ParallelDownloads = 5/' /etc/pacman.conf
# Base system packages
basestrap /mnt base base-devel runit elogind-runit linux-firmware btrfs-progs zstd \
    neovim micro zsh libvirt qemu docker git man chrony-runit cronie-runit
# Kernel - you can choose any (linux linux-lts linux-hardened linux-zen etc...)
basestrap /mnt linux-zen linux-zen-headers
# CPU Microcode (replace with intel-ucode if using intel CPU)
basestrap /mnt amd-ucode
# AMD GPU drivers
basestrap /mnt amdgpu mesa vulkan-radeon libva-mesa-driver mesa-vdpau xf86-video-amdgpu
# Nvidia GPU drivers
# Replace nvidia-dkms with:
# nvidia      - if using linux kernel
# nvidia-lts  - if using linux-lts kernel
# There is also lib32-nvidia-utils and lib32-nvidia-libgl packages 
# available in lib32/multilib repos. Also if you choose to install 
# other init system, there is nvidia-utils-initsystem available, 
# for example nvidia-utils-openrc
basestrap /mnt nvidia-dkms nvidia-settings nvidia-utils nvidia-libgl
# Arch repositories support
basestrap /mnt archlinux-mirrorlist artix-archlinux-support
# Bootloader, AUR packages and packages that are not in 
# Artix repositories will be installed later on

# generate the fstab
fstabgen -U /mnt > /mnt/etc/fstab

# add /tmp as tmpfs in fstab
cat << EOF >> /etc/fstab
tmpfs	/tmp	tmpfs	rw,nosuid,nodev,exec,auto,nouser,async,noatime,mode=1777	0 0
EOF

# 6 System Configuration
# chroot into the new system
artix-chroot /mnt

# - set locale
export LANG="en_GB.UTF-8"
export LC_COLLATE="C"
echo 'LANG="$LANG"' > /etc/locale.conf
echo "KEYMAP=us" > /etc/vconsole.conf
echo "en_GB.UTF-8 UTF-8" > /etc/locale.gen
locale-gen
# - set root password 
passwd
# Replace username with the name for your new user
export USER=username
# Replace hostname with the name for your host
export HOST=hostname
# Replace Europe/London with your Region/City
export TZ="Europe/London"
# - set timezone
ln -sf /usr/share/zoneinfo/$TZ /etc/localtime
hwclock --systohc
# - set hostname
echo $HOST > /etc/hostname
# - set hosts
cat << EOF >> /etc/hosts
# 		
127.0.0.1	localhost
::1		localhost
127.0.1.1	$HOST.localdomain	$HOST
EOF
# - add user 
useradd -mg users -G wheel,storage,power,docker,libvirt,kvm -s /bin/zsh $USER
passwd $USER
echo "$USER ALL=(ALL) ALL" >> /etc/sudoers 
# Set 0 to always ask sudo password .
echo "Defaults timestamp_timeout=300" >> /etc/sudoers
# - Preventing snapshot slowdowns
echo 'PRUNENAMES = ".snapshots"' >> /etc/updatedb.conf

# 6 - fix the mkinitcpio.conf to contain what we actually need.
sed -i 's/^BINARIES=()/BINARIES=("\/usr\/bin\/btrfs")/' /etc/mkinitcpio.conf
# If using amdgpu and would like earlykms
# sed -i 's/MODULES=()/MODULES=(amdgpu)/' /etc/mkinitcpio.conf
# If using nvidia and would like earlykms
# sed -i 's/MODULES=()/MODULES=(nvidia nvidia_modeset nvidia_uvm nvidia_drm)/' /etc/mkinitcpio.conf
sed -i 's/^#COMPRESSION="lz4"/COMPRESSION="lz4"/' /etc/mkinitcpio.conf
sed -i 's/^#COMPRESSION_OPTIONS=()/COMPRESSION_OPTIONS=(-9)/' /etc/mkinitcpio.conf
# if you have more than 1 btrfs drive
# sed -i 's/^HOOKS/HOOKS=(base udev autodetect modconf block resume btrfs filesystems keyboard fsck)/' /etc/mkinitcpio.conf
# else
sed -i 's/^HOOKS.*/HOOKS=(base udev autodetect modconf block resume filesystems keyboard fsck)/' /etc/mkinitcpio.conf

# Replace with your kernel
mkinitcpio -p linux-zen

# 7 Add Arch repositories
cat << EOF >> /etc/pacman.conf
[lib32]
Include = /etc/pacman.d/mirrorlist

# unofficial artix repos
[universe]
Server = https://universe.artixlinux.org/\$arch
Server = https://mirror1.artixlinux.org/universe/\$arch
Server = https://mirror.pascalpuffke.de/artix-universe/\$arch
Server = https://artixlinux.qontinuum.space:4443/universe/os/\$arch

[omniverse]
Server = http://omniverse.artixlinux.org/\$arch

# stable arch repos
[extra]
Include = /etc/pacman.d/mirrorlist-arch

[multilib]
Include = /etc/pacman.d/mirrorlist-arch

[community]
Include = /etc/pacman.d/mirrorlist-arch
EOF

pacman-key --populate archlinux

# Misc options
sed -i 's/^#UseSyslog/UseSyslog/' /etc/pacman.conf
sed -i 's/^#Color/Color\nILoveCandy/' /etc/pacman.conf
sed -i 's/^#TotalDownload/TotalDownload/' /etc/pacman.conf
sed -i 's/^#CheckSpace/CheckSpace/' /etc/pacman.conf
sed -i 's/^#ParallelDownloads.*/ParallelDownloads = 5/' /etc/pacman.conf

# 8 Optimize Makepkg
# Install compressors and bootloader
pacman -Syy pigz pbzip2 refind

# Configure Makepkg
# Use these commands to configure /etc/makepkg.conf
perl -i -0777 -pe 's/^CFLAGS=".*?"/CFLAGS="-march=native -mtune=native -O2 -pipe -fstack-protector-strong --param=ssp-buffer-size=4 -fno-plt"/sm' /etc/makepkg.conf
sed -i 's/^CXXFLAGS.*/CXXFLAGS="\$CFLAGS"/' /etc/makepkg.conf
sed -i 's/^#RUSTFLAGS.*/RUSTFLAGS="-C opt-level=2 -C target-cpu=native"/' /etc/makepkg.conf
sed -i 's/^#BUILDDIR.*/BUILDDIR=\/tmp\/makepkg/' /etc/makepkg.conf
sed -i "s/^#MAKEFLAGS.*/MAKEFLAGS=\"-j$(getconf _NPROCESSORS_ONLN) --quiet\"/" /etc/makepkg.conf
sed -i 's/^COMPRESSGZ=.*/COMPRESSGZ=(pigz -c -f -n)/' /etc/makepkg.conf
sed -i 's/^COMPRESSBZ2=.*/COMPRESSBZ2=(pbzip2 -c -f)/' /etc/makepkg.conf
sed -i "s/^COMPRESSXZ=.*/COMPRESSXZ=(xz -T \"$(getconf _NPROCESSORS_ONLN)\" -c -z --best -)/" /etc/makepkg.conf
sed -i 's/^COMPRESSZST=.*/COMPRESSZST=(zstd -c -z -q --ultra -T0 -22 -)/' /etc/makepkg.conf
sed -i 's/^COMPRESSLRZ=.*/COMPRESSLRZ=(lrzip -9 -q)/' /etc/makepkg.conf
sed -i 's/^COMPRESSLZO=.*/COMPRESSLZO=(lzop -q --best)/' /etc/makepkg.conf
sed -i 's/^COMPRESSZ=.*/COMPRESSZ=(compress -c -f)/' /etc/makepkg.conf
sed -i 's/^COMPRESSLZ4=.*/COMPRESSLZ4=(lz4 -q --best)/' /etc/makepkg.conf
sed -i 's/^COMPRESSLZ=.*/COMPRESSLZ=(lzip -c -f)/' /etc/makepkg.conf
# Here is what it should look like
# Instead of $() statements there should be an output of the commands inside
'
CFLAGS="-march=native -mtune=native -O2 -pipe -fstack-protector-strong --param=ssp-buffer-size=4 -fno-plt"
CXXFLAGS="$CFLAGS"
RUSTFLAGS="-C opt-level=2 -C target-cpu=native"
BUILDDIR=/tmp/makepkg
MAKEFLAGS="-j$(getconf _NPROCESSORS_ONLN) --quiet"
COMPRESSGZ=(pigz -c -f -n)
COMPRESSBZ2=(pbzip2 -c -f)
COMPRESSXZ=(xz -T "$(getconf _NPROCESSORS_ONLN)" -c -z --best -)
COMPRESSZST=(zstd -c -z -q --ultra -T0 -22 -)
COMPRESSLRZ=(lrzip -9 -q)
COMPRESSLZO=(lzop -q --best)
COMPRESSZ=(compress -c -f)
COMPRESSLZ4=(lz4 -q --best)
COMPRESSLZ=(lzip -c -f)
'

# 10 Install bootloader
refind-install 

# 11 Pacman Hooks
mkdir /etc/pacman.d/hooks

cat << EOF > /etc/pacman.d/hooks/refind.hook
[Trigger]
Operation=Upgrade
Type=Package
Target=refind

[Action]
Description = Updating rEFInd on ESP
When=PostTransaction
Exec=/usr/bin/refind-install 
EOF

cat << EOF > /etc/pacman.d/hooks/zsh.hook
[Trigger]
Operation=Install
Operation=Upgrade
Operation=Remove
Type=Path
Target=/usr/bin/*

[Action]
Depends = zsh
When=PostTransaction
Exec=/usr/bin/install -Dm644 /dev/null /var/cache/zsh/pacman
EOF

# If using Nvidia GPU
# Change the linux part in Target and Exec lines if a different kernel is used.
# Also change the nvidia-dkms part if using a different kernel.
cat << EOF > /etc/pacman.d/hooks/nvidia.hook
[Trigger]
Operation=Install
Operation=Upgrade
Operation=Remove
Type=Package
Target=nvidia-dkms
Target=linux-zen

[Action]
Description=Update Nvidia module in initcpio
Depends=mkinitcpio
When=PostTransaction
NeedsTargets
Exec=/bin/sh -c 'while read -r trg; do case \$trg in linux-zen) exit 0; esac; done; /usr/bin/mkinitcpio linux-zen'
EOF

cat << EOF > /etc/udev/rules.d/60-ioschedulers.rules
# set scheduler for NVMe
ACTION=="add|change", KERNEL=="nvme[0-9]*", ATTR{queue/scheduler}="none"
# set scheduler for SSD and eMMC
ACTION=="add|change", KERNEL=="sd[a-z]|mmcblk[0-9]*", ATTR{queue/rotational}=="0", ATTR{queue/scheduler}="mq-deadline"
# set scheduler for rotating disks
ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/rotational}=="1", ATTR{queue/scheduler}="bfq"
EOF

# 12 Configure bootloader
# Replace 1920 1080 with your monitors resolution
sed -i 's/^#resolution 3/resolution 1920 1080/' /boot/EFI/refind/refind.conf
sed -i 's/^#use_graphics_for.*/use_graphics_for linux/' /boot/EFI/refind/refind.conf
sed -i 's/^#scanfor.*/scanfor manual,external,internal/' /boot/EFI/refind/refind.conf
sed -i 's/^#dont_scan_files.*/dont_scan_files vmlinuz-linux-zen/' /boot/EFI/refind/refind.conf

# add the PARTUUID of the root partition to the "root=" option (example below)
cat << EOF >> /boot/EFI/refind/refind.conf
menuentry "Artix Linux" {
    icon     /EFI/refind/icons/os_arch.png
    volume   ROOT
    loader   /vmlinuz-linux-zen
    initrd   /initramfs-linux-zen.img
    options  "root=PARTUUID=$(blkid -s PARTUUID -o value /dev/sda2) rootflags=subvol=@artix rw quiet console=tty2 nmi_watchdog=0 add_efi_memmap initrd=\amd-ucode.img"
    submenuentry "Boot - fallback" {
        initrd /boot/initramfs-linux-zen-fallback.img
    }
}
EOF

# 13 Continue installing packages
su $USER
cd ~
git clone https://aur.archlinux.org/paru.git
cd paru
makepkg -si
cd .. 
sudo rm -dR paru

# At this point you may install any packages you need from AUR or official repositories.
sudo pacman -S snapper networkmanager-runit network-manager-applet openssh-runit \
    zsh-autosuggestions zsh-history-substring-search zsh-syntax-highlighting \
    libimobiledevice usbutils xorg xorg-xinit awesome pipewire pipewire-alsa \
    pipewire-pulse pipewire-jack pipewire-media-session pasystray pavucontrol \
sudo paru -S kitty-git dashbinsh

# If you use a bare git to store dotfiles install them now
# git clone --bare https://github.com/user/repo.git $HOME/.repo
exit

# 14 - reboot into your new install
exit
swapoff -a
umount -R /mnt
reboot

# 15 - After instalation
sudo ln -s /etc/runit/sv/{NetworkManager,sshd,chrony,cronie} /etc/runit/runsvdir/default
sudo sv start NetworkManager sshd chrony cronie
sudo umount /.snapshots
sudo rm -r /.snapshots
sudo snapper -c root create-config /
sudo mount -a
sudo chmod 750 -R /.snapshots
