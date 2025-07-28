#date: 2025-07-28T17:17:18Z
#url: https://api.github.com/gists/8fdeec8ad261728a958dd2b12de788be
#owner: https://api.github.com/users/haroldcris

#!/bin/bash

# MSSQL 2022 Proxmox VM Creator Script
# Version: 1.0

# VM Configuration
VMID=9000
VM_NAME="mssql2022"
ISO_STORAGE="local"  # Change if ISO is stored elsewhere
ISO_IMAGE="ubuntu-22.04-live-server-amd64.iso"
VM_STORAGE="local-lvm"
DISK_SIZE="40G"
RAM="8192"
CORES="4"
BRIDGE="vmbr0"
NET_MODEL="virtio"

echo "====[ Creating SQL Server 2022 VM on Proxmox ]===="

# Step 1: Download ISO if not found
if [ ! -f /var/lib/vz/template/iso/${ISO_IMAGE} ]; then
    echo "Ubuntu ISO not found. Downloading..."
    wget -O /var/lib/vz/template/iso/${ISO_IMAGE} \
        https://releases.ubuntu.com/22.04/${ISO_IMAGE}
else
    echo "ISO already exists."
fi

# Step 2: Create the VM
echo "Creating VM ID $VMID named $VM_NAME..."
qm create $VMID \
  --name $VM_NAME \
  --memory $RAM \
  --cores $CORES \
  --net0 model=$NET_MODEL,bridge=$BRIDGE \
  --ide2 $ISO_STORAGE:iso/${ISO_IMAGE},media=cdrom \
  --boot order=ide2 \
  --scsihw virtio-scsi-pci \
  --scsi0 $VM_STORAGE:${DISK_SIZE} \
  --ostype l26 \
  --agent 1

# Step 3: Enable QEMU guest agent and serial console
qm set $VMID --serial0 socket --vga serial0

# Step 4: Start the VM
qm start $VMID

echo ""
echo "VM $VMID ($VM_NAME) created and started."
echo "Please complete Ubuntu 22.04 installation manually via the Proxmox console."
echo ""
echo "Afterward, run the SQL Server setup script inside the VM to install MSSQL 2022."
