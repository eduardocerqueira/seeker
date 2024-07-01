#date: 2024-07-01T16:38:47Z
#url: https://api.github.com/gists/8d8136d45fd3f623b10895721f3916ce
#owner: https://api.github.com/users/acro5piano

# Pre installtion notice: 
# - The iso image can be downloaed from: https://www.android-x86.org/download.html
# - Linux Zen is recommended. See: https://riq0h.jp/2020/12/07/210053/
# - Also see: https://wiki.archlinux.org/title/QEMU

sudo pacman -S qemu-full
sudo pacman -S qemu-desktop

qemu-img create -f raw android 16G

# -enable-kvm is the point. Otherwise, Android is not starting.
qemu-system-x86_64 -enable-kvm -cdrom ~/Downloads/android-x86_64-9.0-r2-k49.iso -boot menu=on -drive file=./android,format=raw -m 2G
