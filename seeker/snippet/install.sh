#date: 2022-07-14T16:59:34Z
#url: https://api.github.com/gists/c501d61fcf63168f8a6973088845a99c
#owner: https://api.github.com/users/Huangxt57

#!/usr/bin/bash

user=xxx

# Test whether virtualization is supported for CPU
vm_support=$(egrep '(vmx|svm)' /proc/cpuinfo)
if [ "$a" == "" ]; then
  echo "Virtualization is not supported, quit..."
  exit 1
fi

# Load kvm kernel module (if necessary) on AMD machines
kvm_load=$(lsmod | grep kvm)
if [ "$kvm_load" == "" ]; then
  modprobe kvm
  modprobe kvm_amd
fi

# Install KVM on Debian
# https://wiki.debian.org/KVM#Installation
apt update
apt install -y --no-install-recommends \
  qemu-utils \
  qemu-system \
  qemu-kvm \
  libvirt-daemon-system \
  libvirt-clients \
  bridge-utils

# Check installation
echo q | systemctl status libvirtd

# Add user to allow using in non-privilige mode
adduser $user libvirt
id $user

# See vm list
virsh --version
virsh list