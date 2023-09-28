#date: 2023-09-28T17:02:20Z
#url: https://api.github.com/gists/c5b71d6addf36974763925cb598b504d
#owner: https://api.github.com/users/iotku

#!/bin/bash
set -x

# Stop display manager
#systemctl --user -M user@ stop pipewire.service pipewire.socket
systemctl --user -M user@ stop plasma*
systemctl stop display-manager

# Unbind VTconsoles: might not be needed
echo 0 > /sys/class/vtconsole/vtcon0/bind
echo 0 > /sys/class/vtconsole/vtcon1/bind

# Unbind EFI Framebuffer
#echo efi-framebuffer.0 > /sys/bus/platform/drivers/efi-framebuffer/unbind

# Unload NVIDIA kernel modules (try REALLY hard with needless sleeps)
sleep 1
modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia
sleep 2
modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia
sleep 1
# Detach GPU devices from host
# Use your GPU and HDMI Audio PCI host device
virsh nodedev-detach pci_0000_0c_00_0
virsh nodedev-detach pci_0000_0c_00_1

# Load vfio module
modprobe vfio-pci
