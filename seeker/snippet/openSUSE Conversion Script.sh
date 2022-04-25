#date: 2022-04-25T17:18:36Z
#url: https://api.github.com/gists/bd29113383f78557eef465691f21b2f9
#owner: https://api.github.com/users/0x1a8510f2

#!/bin/bash
set -xe
: ${repo:=https://download.opensuse.org/distribution/openSUSE-stable/repo/oss}
#: ${repo:=https://download.opensuse.org/tumbleweed/repo/oss}
: ${arch:=$(uname -m)}
: ${vncpassword:=supercomplexpassword}
#: ${append:=vnc=1 vncpassword=$vncpassword}
: ${append:=ssh=1 sshpassword=$vncpassword}
#append+=" addon=https://download.opensuse.org/update/openSUSE-stable/"
#append+=" autoyast=https://www.zq1.de/~bernhard/linux/opensuse/autoyast.leap152.xml"
pkgs="wget kexec-tools"
zypper -n install $pkgs ||
  DEBIAN_FRONTEND=noninteractive apt-get -y install $pkgs ||
  dnf install -y $pkgs ||
  pacman --noconfirm -S $pkgs ||
  emerge $pkgs ||
  true
which wget
which kexec
mkdir -p /dev/shm/
mount -t tmpfs tmpfs /dev/shm
cd /dev/shm/
wget $repo/boot/$arch/loader/{linux,initrd}
kexec -l linux --initrd=initrd --reset-vga --append="install=$repo $append"
sync ; echo u > /proc/sysrq-trigger ; sync
kexec -e