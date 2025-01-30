#date: 2025-01-30T16:49:58Z
#url: https://api.github.com/gists/1f755fe64ab707ca65218118df56025e
#owner: https://api.github.com/users/bsautner

dpkg --add-architecture i386

apt update && sudo apt upgrade -y
uname -r
ls /usr/src/linux-headers-$(uname -r)

apt install --reinstall build-essential dkms linux-headers-$(uname -r) wget -y
sudo bash -c "echo 'blacklist nouveau' > /etc/modprobe.d/blacklist-nouveau.conf"
sudo bash -c "echo 'options nouveau modeset=0' >> /etc/modprobe.d/blacklist-nouveau.conf"
sudo update-initramfs -u
apt install libc6:i386 libvulkan1
sudo rm -rf /lib/modules/$(uname -r)/build/* && sudo apt install --reinstall linux-headers-$(uname -r)

#reboot