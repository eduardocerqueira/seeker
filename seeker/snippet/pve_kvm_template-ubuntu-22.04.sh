#date: 2022-12-29T16:49:07Z
#url: https://api.github.com/gists/b0b527723432c687b4f928a59c3c8052
#owner: https://api.github.com/users/vdrizheruk

#!/bin/bash
set -e
#set -x

VM_BRIDGE=${VM_BRIDGE:-vmbr24}
VM_ID=9001
VM_NAME="linux-ubuntu-server-22.04"
IMAGE_PATH=$(realpath ~/jammy-server-cloudimg-amd64.img)
VM_STORAGE="local"

echo "==> Destroy previous vm if exists"
qm list | grep ${VM_ID} | awk '{ print $1 }' | xargs --no-run-if-empty qm destroy

echo "==> Creating a new vm (${VM_ID})"
qm create ${VM_ID} --name ${VM_NAME} --memory 1024 --net0 virtio,bridge=${VM_BRIDGE}


echo "==> Importing the downloaded disk to ${VM_STORAGE} storage"
qm importdisk ${VM_ID} ${IMAGE_PATH} ${VM_STORAGE}

echo "==> Attach the new disk to the VM as scsi drive"
qm set ${VM_ID} --scsihw virtio-scsi-pci --scsi0 ${VM_STORAGE}:${VM_ID}/vm-${VM_ID}-disk-0.raw

echo "==> Adding Cloud-Init CDROM drive"
qm set ${VM_ID} --ide2 ${VM_STORAGE}:cloudinit
qm set ${VM_ID} --boot c --bootdisk scsi0

echo "==> Configuring a serial console and use it as a display."
qm set ${VM_ID} --serial0 socket --vga serial0

echo "==> Configuring dhcp."
qm set ${VM_ID} --ipconfig0 ip=dhcp,ip6=dhcp

echo "==> Convert vm into to a template"
qm template ${VM_ID}