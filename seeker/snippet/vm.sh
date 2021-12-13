#date: 2021-12-13T17:02:27Z
#url: https://api.github.com/gists/0aa6d17b0275e6bedb8e0b899c2b3073
#owner: https://api.github.com/users/drgmr

# Create a new disk file
# qemu-img create -f qcow2 vm.img 100G

# Boot from a distro installer .iso
# qemu-system-x86_64 -M accel=hvf --cpu host -hda vm.img -cdrom ~/Downloads/$DISTRO.iso -boot d -m 2048

# Start the VM
qemu-system-x86_64 -M accel=hvf --cpu host vm.img -m 2048 -net user,hostfwd=tcp::2222-:22 -net nic,model=virtio
