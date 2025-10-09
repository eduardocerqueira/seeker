#date: 2025-10-09T16:50:33Z
#url: https://api.github.com/gists/99c5efca589cc83a9b89666a662861ed
#owner: https://api.github.com/users/MrZoidberg

set -euo pipefail

get_storage_type() {
  local id="$1"
  # Prefer config file: exact and quiet
  local t
  t="$(awk -v id="$id" -F'[: ]+' '
      $1 ~ /^(dir|zfspool|zfs|lvmthin|lvm|rbd|cephfs|nfs|cifs)$/ && $2==id { print $1; found=1; exit }
      END { if (!found) exit 1 }
    ' /etc/pve/storage.cfg 2>/dev/null)" || true
  if [[ -n "$t" ]]; then
    printf '%s\n' "$t"
    return 0
  fi
  # Fallback to pvesm status; ignore stderr noise from broken storages
  pvesm status 2>/dev/null | awk -v s="$id" '$1==s{print $2; exit}'
}

# ====== DEFAULTS (adjust if needed) ======
DISK_STORE="local"                  # VM disk storage (where VM disks + CI drive live)
ISO_STORE="local"                      # where the cloud image file resides
CLOUDIMG="ubuntu-24.04-server-cloudimg-amd64.img"
RAM_MB=1024                           # 10 GB
VCPUS=1
SSH_PUBKEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKz37q3ePf//WaIA41DUvTIcl6nmfpA2wmHnxXsBH4Ls'
CI_USER="mike"
NET0_BR="vmbr0"                        # DHCP v4 (primary/default gw)
NET1_BR="vnet1"                        # DHCP v4
DISK_SIZE="4G"
# ========================================

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <vmname> <vmid>"
  exit 1
fi
VMNAME="$1"
VMID="$2"

STYPE="$(get_storage_type "${DISK_STORE}")"
if [[ -z "${STYPE}" ]]; then
  echo "ERROR: could not determine storage type for '${DISK_STORE}'"
  exit 1
fi
echo "Storage '${DISK_STORE}' type: ${STYPE}"

# Pick format + disk opts based on storage type
case "${STYPE}" in
  zfs|zfspool)  IMG_FORMAT="qcow2";   DISK_OPTS="ssd=1,discard=on,backup=1" ;;
  lvm|lvmthin)  IMG_FORMAT="qcow2";   DISK_OPTS="discard=on,backup=1" ;;
  dir)          IMG_FORMAT="qcow2"; DISK_OPTS="discard=on,backup=1" ;;
  *)            IMG_FORMAT="qcow2"; DISK_OPTS="backup=1" ;;
esac

# Resolve image path (common locations)
IMG_PATH=""
for p in "./${CLOUDIMG}" "/var/lib/vz/template/iso/${CLOUDIMG}" "/var/lib/vz/template/qemu/${CLOUDIMG}"; do
  [[ -f "$p" ]] && IMG_PATH="$p" && break
done
if [[ -z "$IMG_PATH" ]]; then
  echo "ERROR: Cannot find ${CLOUDIMG}. Put it next to this script or in /var/lib/vz/template/iso/"
  exit 1
fi


# Ensure 'local' storage supports snippets and the dir exists
# pvesm set ${DISK_STORE} --content iso,vztmpl,backup,images,snippets >/dev/null 2>&1 || true
mkdir -p /var/lib/vz/snippets

# 1) Generate a fresh random password for ${CI_USER} (printed at end)
RAND_PW="$(openssl rand -base64 18 | tr -d '\n' | sed 's/[[:space:]]//g')"

# 2) Paths for snippets (user-data + meta)
CICUSTOM_DIR="/var/lib/vz/snippets"
USERDATA="${CICUSTOM_DIR}/${VMNAME}-${VMID}-cloudinit.yaml"
METADATA="${CICUSTOM_DIR}/${VMNAME}-${VMID}-meta.yaml"

# 3) Write USER-DATA (no host/guest var ambiguity; only ${VMNAME}, ${CI_USER}, ${RAND_PW}, ${SSH_PUBKEY} expand here)
cat > "${USERDATA}" <<EOF
#cloud-config
manage_etc_hosts: true
package_update: false
package_upgrade: false

ssh_pwauth: true
disable_root: true

preserve_hostname: false
hostname: ${VMNAME}

users:
  - default
  - name: ${CI_USER}
    gecos: ${CI_USER}
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    groups: sudo, docker
    lock_passwd: false
    ssh_authorized_keys:
      - ${SSH_PUBKEY}

chpasswd:
  expire: false
  list:
    - ${CI_USER}:${RAND_PW}

write_files:
  # Force apt to use IPv4 (prevents IPv6 mirror failures)
  - path: /etc/apt/apt.conf.d/99force-ipv4
    owner: root:root
    permissions: '0644'
    content: |
      Acquire::ForceIPv4 "true";

  # sshd: "**********"
  - path: "**********"
    owner: root:root
    permissions: '0644'
    content: |
      PasswordAuthentication yes

  # Netplan overlay: default GW on eth0; suppress DHCP routes from eth1
  - path: /etc/netplan/90-ci-default-gw.yaml
    owner: root:root
    permissions: '0600'
    content: |
      network:
        version: 2
        renderer: networkd
        ethernets:
          eth0:
            dhcp4: true
            dhcp6: false
            routes:
              - to: default
                via: 192.168.10.1
                metric: 10
          eth1:
            dhcp4: true
            dhcp6: false
            dhcp4-overrides:
              use-routes: false

runcmd:
  # Apply netplan first and wait for eth0 connectivity
  - netplan apply
  - systemctl enable systemd-networkd-wait-online.service || true
  - systemd-networkd-wait-online -i eth0 --timeout=30 || true

  # ---- APT (IPv4-only) ----
  - apt-get update
  - apt-get upgrade -y

  # ---- Docker CE (official repo, per docs.docker.com) ----
  - apt-get install -y ca-certificates curl gnupg
  - install -m 0755 -d /etc/apt/keyrings
  - curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  - chmod a+r /etc/apt/keyrings/docker.gpg
  - bash -lc 'echo "deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \$(. /etc/os-release && echo \$VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list'
  - apt-get update
  - apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  - usermod -aG docker ${CI_USER}
  - systemctl enable docker || true
  - systemctl start docker || true

  # ---- Other packages ----
  - apt-get install -y qemu-guest-agent btop net-tools
  - systemctl enable qemu-guest-agent || true
  - systemctl start qemu-guest-agent || true

  # ---- Micro editor ----
  - bash -lc 'cd /root && curl -fsSL https://getmic.ro | bash && install -m 0755 micro /usr/local/bin/micro'

  # ---- SSH service ----
  - systemctl enable ssh
  - systemctl restart ssh

  # One-time reboot to settle everything
  - reboot
EOF

# 4) Write META-DATA with a fresh instance-id (forces cloud-init re-apply)
cat > "${METADATA}" <<EOF
instance-id: vm-${VMID}-$(date +%s)
local-hostname: ${VMNAME}
EOF

# Sanity: verify snippets exist before calling qm set (prevents "volume does not exist")
if [[ ! -s "${USERDATA}" ]]; then
  echo "ERROR: user-data snippet not found or empty at ${USERDATA}"
  exit 1
fi
if [[ ! -s "${METADATA}" ]]; then
  echo "ERROR: meta-data snippet not found or empty at ${METADATA}"
  exit 1
fi

# 5) Create or update the VM
if ! qm status "${VMID}" >/dev/null 2>&1; then
  echo "Creating VM ${VMID} (${VMNAME})..."
  qm create "${VMID}" \
    --name "${VMNAME}" \
    --memory "${RAM_MB}" \
    --cores "${VCPUS}" \
    --net0 "virtio,bridge=${NET0_BR}" \
    --net1 "virtio,bridge=${NET1_BR}" \
    --agent "enabled=1,fstrim_cloned_disks=1"

  echo "Importing cloud image to ${DISK_STORE}..."
  qm importdisk "${VMID}" "${IMG_PATH}" "${DISK_STORE}" --format qcow2

  VOLID="$(qm config "${VMID}" | sed -n 's/^scsi0:\s\+\([^,]\+\).*/\1/p')"
  echo "Volume ID ${VOLID}"

  echo "Attaching new disk "
  qm set "${VMID}" --scsihw virtio-scsi-pci --scsi0 "${DISK_STORE}:0,${DISK_OPTS},import-from=${IMG_PATH},format=${IMG_FORMAT}"

  echo "Resizing disk to ${DISK_SIZE}..."
  qm resize "${VMID}" scsi0 "${DISK_SIZE}"

  echo "Adding Cloud-Init drive (ide2 on ${DISK_STORE})..."
  qm set "${VMID}" --ide2 "${DISK_STORE}:cloudinit"

  # Serial console (cloud images expect it)
  qm set "${VMID}" --serial0 socket --vga serial0

  # Boot from the imported cloud image
  qm set "${VMID}" --boot order=scsi0 --bootdisk scsi0

  # Attach snippets (user + meta)
  qm set "${VMID}" \
    --ciuser "${CI_USER}" \
    --ipconfig0 "ip=dhcp" \
    --ipconfig1 "ip=dhcp" \
    --cicustom "user=local:snippets/$(basename "${USERDATA}"),meta=local:snippets/$(basename "${METADATA}")"

  qm set "${VMID}" --ostype l26 --onboot 1
else
  echo "VM ${VMID} already exists — updating Cloud-Init configuration..."
  # Ensure it has a cloud-init drive; if not, add one
  if ! qm config "${VMID}" | grep -q '^ide2: .*cloudinit'; then
    qm set "${VMID}" --ide2 "${DISK_STORE}:cloudinit"
  fi
  # Ensure the VM uses our snippets (user + meta)
  qm set "${VMID}" \
    --cicustom "user=local:snippets/$(basename "${USERDATA}"),meta=local:snippets/$(basename "${METADATA}")"
fi

# 6) Rebuild Cloud-Init ISO to include the updated snippets
echo "Regenerating Cloud-Init ISO..."
qm cloudinit update "${VMID}"

# 7) Show effective user-data/meta-data (sanity)
echo "Effective user-data:"
qm cloudinit dump "${VMID}" user || true
echo "Effective meta-data:"
qm cloudinit dump "${VMID}" meta || true

# 8) Start (or restart) the VM
if qm status "${VMID}" 2>/dev/null | grep -q running; then
  echo "VM ${VMID} is running. Rebooting to apply new cloud-init (guest will reboot again via runcmd)..."
  qm reboot "${VMID}" || true
else
  echo "Starting VM ${VMID}..."
  qm start "${VMID}"
fi

echo
echo "✅ VM ${VMID} (${VMNAME}) provision/update triggered."
echo "   - User: ${CI_USER}"
echo "   - Temporary password: "**********"
echo "   - Docker CE + compose (official repo), qemu-guest-agent, btop, net-tools, micro"
echo "   - Netplan: default via 192.168.10.1 on eth0; routes from eth1 suppressed"
echo "   - Cloud-Init ISO rebuilt; instance-id rotated."
echo "   - Expect one additional reboot from 'runcmd'."d."
echo "   - Expect one additional reboot from 'runcmd'."