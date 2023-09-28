#date: 2023-09-28T17:02:20Z
#url: https://api.github.com/gists/c5b71d6addf36974763925cb598b504d
#owner: https://api.github.com/users/iotku

#!/bin/bash
set -x

# Attach GPU devices to host
# Use your GPU and HDMI Audio PCI host device
virsh nodedev-reattach pci_0000_0c_00_0
virsh nodedev-reattach pci_0000_0c_00_1

# Unload vfio module
modprobe -r vfio-pci

# Load AMD kernel module
#modprobe amdgpu

# Rebind framebuffer to host
#echo "efi-framebuffer.0" > /sys/bus/platform/drivers/efi-framebuffer/bind

# Load NVIDIA kernel modules
modprobe nvidia_drm
modprobe nvidia_modeset
modprobe nvidia_uvm
modprobe nvidia

# Bind VTconsoles: might not be needed
#echo 1 > /sys/class/vtconsole/vtcon0/bind
#echo 1 > /sys/class/vtconsole/vtcon1/bind

# Restart Display Manager
systemctl start display-manager
