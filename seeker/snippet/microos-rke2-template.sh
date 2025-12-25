#date: 2025-12-25T16:57:29Z
#url: https://api.github.com/gists/04003b1cb8e52ed6460fc589dc9639e3
#owner: https://api.github.com/users/igor-podpalchenko

#!/usr/bin/env bash
set -euo pipefail

# microos-rke2-template.sh
# Prepare openSUSE MicroOS VMware image for Rancher (RKE2) provisioning on vSphere
#
# Usage:
#   sudo bash microos-rke2-template.sh --stage1
#   # VM reboots automatically
#   sudo bash microos-rke2-template.sh --post
#
# Optional:
#   sudo bash microos-rke2-template.sh --install-rke2   (install RKE2 binaries via get.rke2.io)
#   sudo bash microos-rke2-template.sh --finalize       (cloud-init clean + machine-id reset + shutdown)
#
# Notes:
# - Uses transactional-update (MicroOS) for package installs, then reboots.
# - Forces cloud-init to run full pipeline and prefer NoCloud (Rancher config-drive).
# - Disables Combustion (Ignition-like first boot).
# - Avoids enabling/starting rke2-server in template (Rancher will do that).
# - Suppresses fd0 noise via GRUB kernel cmdline (cloud-init-level effect: takes effect after reboot).
# - Patches GRUB timeout from 10s to 5s.

log() { echo "[$(date -Is)] $*"; }

need_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run as root (sudo)." >&2
    exit 1
  fi
}

tu() {
  # transactional-update is typically in /usr/sbin (may not be in PATH for non-root)
  if [[ -x /usr/sbin/transactional-update ]]; then
    /usr/sbin/transactional-update "$@"
  else
    transactional-update "$@"
  fi
}

patch_grub() {
  # Patch GRUB timeout and add modprobe blacklist for floppy.
  # This avoids OS-level blacklisting and instead changes kernel cmdline at boot.
  local grub_def="/etc/default/grub"
  [[ -f "$grub_def" ]] || { log "GRUB defaults not found ($grub_def); skipping GRUB patch"; return 0; }

  log "Patching GRUB: timeout=5, add modprobe.blacklist=floppy"

  # timeout -> 5
  if grep -q '^GRUB_TIMEOUT=' "$grub_def"; then
    sed -i 's/^GRUB_TIMEOUT=.*/GRUB_TIMEOUT=5/' "$grub_def"
  else
    echo 'GRUB_TIMEOUT=5' >> "$grub_def"
  fi

  # ensure cmdline contains modprobe.blacklist=floppy (use DEFAULT if present, else LINUX)
  if grep -q '^GRUB_CMDLINE_LINUX_DEFAULT=' "$grub_def"; then
    if ! grep -q 'modprobe\.blacklist=floppy' "$grub_def"; then
      sed -i 's/^\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"/\1 modprobe.blacklist=floppy"/' "$grub_def"
    fi
  elif grep -q '^GRUB_CMDLINE_LINUX=' "$grub_def"; then
    if ! grep -q 'modprobe\.blacklist=floppy' "$grub_def"; then
      sed -i 's/^\(GRUB_CMDLINE_LINUX="[^"]*\)"/\1 modprobe.blacklist=floppy"/' "$grub_def"
    fi
  else
    echo 'GRUB_CMDLINE_LINUX_DEFAULT="modprobe.blacklist=floppy"' >> "$grub_def"
  fi

  # regenerate grub config (BIOS + UEFI if present)
  grub2-mkconfig -o /boot/grub2/grub.cfg || true
  grub2-mkconfig -o /boot/efi/EFI/opensuse/grub.cfg 2>/dev/null || true
}

stage1_install_packages() {
  log "Stage1: transactional package install (will reboot afterwards)"

  # Map your Ubuntu list to SUSE/MicroOS package names.
  # - nfs-common -> nfs-client
  # - openssh-server -> openssh
  # - net-tools -> net-tools-deprecated
  #
  # cloud-guest-utils/cloud-image-utils/cloud-initramfs-growroot are Ubuntu/Debian-specific;
  # for MicroOS we keep cloud-init + xorriso (useful for testing NoCloud) + core tools.

  tu -n pkg install \
    cloud-init \
    open-vm-tools \
    open-iscsi \
    nfs-client \
    curl wget ca-certificates \
    openssh \
    containerd \
    apparmor-parser \
    net-tools-deprecated \
    xorriso

  log "Stage1 done. Rebooting to activate snapshot."
  reboot
}

post_config() {
  log "Post: enable services, disable combustion/ignition, configure rke2 sysctl/modules, cloud-init wiring, GRUB patch"

  # Enable essentials
  systemctl enable --now sshd || true
  systemctl enable --now vmtoolsd || true
  systemctl enable --now vgauthd || true
  systemctl enable --now iscsid || systemctl enable --now iscsid.service || true

  # Disable Combustion (MicroOS first-boot mechanism) + ignition units if present
  systemctl disable --now combustion 2>/dev/null || true
  systemctl mask combustion 2>/dev/null || true
  systemctl disable --now ignition-firstboot-complete.service ignition-disks.service ignition-fetch.service ignition-mount.service ignition.service 2>/dev/null || true
  systemctl mask ignition-firstboot-complete.service ignition-disks.service ignition-fetch.service ignition-mount.service ignition.service 2>/dev/null || true
  rm -rf /var/lib/combustion /etc/combustion /oem 2>/dev/null || true

  # RKE2 baseline kernel modules
  cat > /etc/modules-load.d/rke2.conf <<'EOF'
overlay
br_netfilter
EOF
  modprobe overlay br_netfilter || true

  # RKE2 baseline sysctl
  cat > /etc/sysctl.d/90-rke2.conf <<'EOF'
net.ipv4.ip_forward=1
net.bridge.bridge-nf-call-iptables=1
EOF
  sysctl --system || true

  # crictl config (optional but handy)
  cat > /etc/crictl.yaml <<'EOF'
runtime-endpoint: unix:///run/rke2/containerd/containerd.sock
image-endpoint: unix:///run/rke2/containerd/containerd.sock
timeout: 10
debug: false
EOF

  # Cloud-init: force enable (avoid disabled-by-generator timing issues) + prefer NoCloud
  rm -f /etc/cloud/cloud-init.disabled
  cat > /etc/cloud/ds-identify.cfg <<'EOF'
policy: enabled
EOF
  mkdir -p /etc/cloud/cloud.cfg.d
  cat > /etc/cloud/cloud.cfg.d/90-datasource.cfg <<'EOF'
datasource_list: [ NoCloud, VMware, None ]
EOF

  # Critical: wire the full cloud-init pipeline.
  # MicroOS ships many cloud-init units as "static"; enabling only cloud-init-local runs only init-local.
  mkdir -p /etc/systemd/system/multi-user.target.wants
  ln -sf /usr/lib/systemd/system/cloud-init.target \
    /etc/systemd/system/multi-user.target.wants/cloud-init.target

  mkdir -p /etc/systemd/system/cloud-init.target.wants
  for u in \
    cloud-init-local.service \
    cloud-init-main.service \
    cloud-init-network.service \
    cloud-config.service \
    cloud-final.service \
    cloud-init.service
  do
    if [[ -f "/usr/lib/systemd/system/$u" ]]; then
      ln -sf "/usr/lib/systemd/system/$u" "/etc/systemd/system/cloud-init.target.wants/$u"
    fi
  done
  systemctl daemon-reload

  # Patch GRUB (boot timeout + fd0 suppression at bootloader/kernel-cmdline level)
  patch_grub

  log "Post done."
  log "Cloud-init should now fully run when Rancher attaches NoCloud seed (/dev/sr0)."
  log "fd0 kernel errors will stop after next reboot (GRUB cmdline modprobe.blacklist=floppy)."
  log "Bootloader timeout set to 5s."
}

install_rke2_binaries() {
  log "Installing RKE2 binaries via official installer (does NOT enable/start service)"
  # This is optional: Rancher can install RKE2 itself. But pre-install is OK.
  curl -sfL https://get.rke2.io | sh -
  ls -l /usr/local/bin/rke2 /opt/rke2/bin/rke2 2>/dev/null || true
}

finalize_for_template() {
  log "Finalizing for template: cloud-init clean + machine-id reset + shutdown"

  # Stop cloud-init units if present
  systemctl stop cloud-init cloud-init-local cloud-config cloud-final 2>/dev/null || true

  # Reset cloud-init state for clones
  cloud-init clean --logs --seed || true
  rm -rf /var/lib/cloud/instances /var/lib/cloud/instance

  # Reset machine-id so clones are unique
  truncate -s 0 /etc/machine-id
  rm -f /var/lib/dbus/machine-id
  ln -sf /etc/machine-id /var/lib/dbus/machine-id

  sync
  shutdown -h now
}

usage() {
  cat <<'EOF'
Usage:
  sudo bash microos-rke2-template.sh --stage1
  # reboots automatically
  sudo bash microos-rke2-template.sh --post

Optional:
  sudo bash microos-rke2-template.sh --install-rke2
  sudo bash microos-rke2-template.sh --finalize
EOF
}

main() {
  need_root

  if [[ $# -lt 1 ]]; then
    usage
    exit 2
  fi

  case "$1" in
    --stage1)
      stage1_install_packages
      ;;
    --post)
      post_config
      ;;
    --install-rke2)
      install_rke2_binaries
      ;;
    --finalize)
      finalize_for_template
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"