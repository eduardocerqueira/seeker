#date: 2026-01-08T17:11:52Z
#url: https://api.github.com/gists/b14659e5828231f5d0425a4c8fb9a4b5
#owner: https://api.github.com/users/mpabegg

#!/usr/bin/env bash
set -euo pipefail

# Fedora + Btrfs (/root, /home) + Snapper bootstrapper
# - Verifies Btrfs mounts/subvols
# - Installs snapper + btrfs-assistant + libdnf5-plugin-actions + gum
# - Creates snapper configs for / and /home (idempotent-ish)
# - Fixes SELinux contexts for .snapshots
# - Enables snapper timeline + cleanup timers
# - Disables timeline snapshots for home
# - Sets reasonable retention for root
# - Adds dnf5 actions hooks for root pre/post snapshots

need_cmd() { command -v "$1" >/dev/null 2>&1; }

is_root() { [[ ${EUID:-$(id -u)} -eq 0 ]]; }

# ---------- gum helpers ----------
ginfo()   { gum style --foreground 81  "ℹ $*"; }
gok()     { gum style --foreground 42  "✔ $*"; }
gwarn()   { gum style --foreground 214 "⚠ $*"; }
gerr()    { gum style --foreground 196 "✖ $*"; }
gtitle()  { gum style --bold --foreground 213 "$*"; }

# ---------- preflight ----------
if ! need_cmd dnf; then
  echo "This script expects Fedora/RHEL-like with dnf."
  exit 1
fi

# Ensure gum exists (install early, before fancy UI)
if ! need_cmd gum; then
  echo "[bootstrap] Installing gum (Charm) for UI..."
  if is_root; then
    dnf -y install gum >/dev/null
  else
    sudo dnf -y install gum >/dev/null
  fi
fi

# Re-check now that gum exists
if ! need_cmd gum; then
  echo "Failed to install gum."
  exit 1
fi

gtitle "Fedora Btrfs (/root, /home) + Snapper setup"

# Ask for sudo once if not root
if ! is_root; then
  ginfo "This script needs sudo for installs and system changes."
  sudo -v
fi

run() {
  # run <cmd...> : runs command with sudo if needed; shows spinner
  local -a cmd=( "$@" )
  local label="${cmd[*]}"
  gum spin --spinner dot --title "$label" -- \
    bash -lc "$(is_root && echo "${cmd[*]}" || echo "sudo ${cmd[*]}")"
}

# ---------- checks ----------
gtitle "1) Verifying system state"

if ! need_cmd findmnt; then
  run dnf -y install util-linux
fi

# Verify /root and /home are btrfs
root_fstype="$(findmnt -n -o FSTYPE / || true)"
home_fstype="$(findmnt -n -o FSTYPE /home || true)"

if [[ "$root_fstype" != "btrfs" ]]; then
  gerr "/ is not mounted as btrfs (found: ${root_fstype:-unknown}). Aborting."
  exit 1
fi
if [[ "$home_fstype" != "btrfs" ]]; then
  gerr "/home is not mounted as btrfs (found: ${home_fstype:-unknown}). Aborting."
  exit 1
fi
gok "Root and home are on Btrfs"

# Verify expected subvol names (best-effort, based on mount options)
root_opts="$(findmnt -n -o OPTIONS / || true)"
home_opts="$(findmnt -n -o OPTIONS /home || true)"

expect_root="subvol=/root"
expect_home="subvol=/home"

if ! grep -q "$expect_root" <<<"$root_opts"; then
  gwarn "Mount options for / do not include '${expect_root}'."
  gwarn "Found OPTIONS: $root_opts"
  if ! gum confirm "Continue anyway?"; then
    exit 1
  fi
else
  gok "Detected / mounted with ${expect_root}"
fi

if ! grep -q "$expect_home" <<<"$home_opts"; then
  gwarn "Mount options for /home do not include '${expect_home}'."
  gwarn "Found OPTIONS: $home_opts"
  if ! gum confirm "Continue anyway?"; then
    exit 1
  fi
else
  gok "Detected /home mounted with ${expect_home}"
fi

# ---------- install packages ----------
gtitle "2) Installing packages"
run dnf -y install snapper btrfs-assistant libdnf5-plugin-actions policycoreutils-python-utils

gok "Packages installed"

# ---------- snapper configs ----------
gtitle "3) Configuring Snapper"

# helper: check if config exists
snapper_has_config() {
  local name="$1"
  snapper list-configs 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$name"
}

if snapper_has_config "root"; then
  gok "Snapper config 'root' already exists"
else
  run snapper -c root create-config /
  gok "Created Snapper config 'root' for /"
fi

if snapper_has_config "home"; then
  gok "Snapper config 'home' already exists"
else
  run snapper -c home create-config /home
  gok "Created Snapper config 'home' for /home"
fi

# Ensure snapshots dirs exist
if [[ ! -d "/.snapshots" ]]; then
  gwarn "Expected /.snapshots to exist but it does not. Something is off."
  if ! gum confirm "Continue anyway?"; then
    exit 1
  fi
fi

if [[ ! -d "/home/.snapshots" ]]; then
  gwarn "Expected /home/.snapshots to exist but it does not. Something is off."
  if ! gum confirm "Continue anyway?"; then
    exit 1
  fi
fi

# ---------- SELinux contexts ----------
gtitle "4) Fixing SELinux contexts for .snapshots"
# restorecon is safe to re-run
run restorecon -RFv /.snapshots
run restorecon -RFv /home/.snapshots
gok "SELinux contexts restored"

# ---------- allow user access ----------
gtitle "5) Allowing snapshot inspection for your user"

user_name="${SUDO_USER:-$USER}"
if [[ -z "${user_name:-}" ]]; then
  user_name="$(id -un)"
fi

# Apply ACL sync + allow user on both configs
run snapper -c root set-config "ALLOW_USERS=${user_name}" SYNC_ACL=yes
run snapper -c home set-config "ALLOW_USERS=${user_name}" SYNC_ACL=yes
gok "Configured Snapper user access for '${user_name}'"

# ---------- timers ----------
gtitle "6) Enabling Snapper timers"
run systemctl enable --now snapper-timeline.timer
run systemctl enable --now snapper-cleanup.timer
gok "snapper-timeline + snapper-cleanup timers enabled"

# ---------- policy: disable home timeline ----------
gtitle "7) Snapshot policy"

run snapper -c home set-config TIMELINE_CREATE=no
gok "Disabled timeline snapshots for home"

# Root retention defaults (reasonable desktop settings)
if gum confirm "Set recommended root snapshot retention (hourly=6, daily=7, weekly=2, monthly/yearly=0)?"; then
  run snapper -c root set-config \
    TIMELINE_LIMIT_HOURLY=6 \
    TIMELINE_LIMIT_DAILY=7 \
    TIMELINE_LIMIT_WEEKLY=2 \
    TIMELINE_LIMIT_MONTHLY=0 \
    TIMELINE_LIMIT_YEARLY=0
  gok "Root retention updated"
else
  gwarn "Skipped retention tuning"
fi

# ---------- dnf5 actions integration ----------
gtitle "8) DNF (dnf5) pre/post snapshot integration"

actions_dir="/etc/dnf/libdnf5-plugins/actions.d"
actions_file="${actions_dir}/snapper.actions"
desired_content=$'pre_transaction::::/usr/bin/snapper -c root create -t pre -p -d "dnf pre"\npost_transaction::::/usr/bin/snapper -c root create -t post -p -d "dnf post"\n'

run mkdir -p "$actions_dir"

# Write idempotently (only if changed)
current_content=""
if [[ -f "$actions_file" ]]; then
  current_content="$(cat "$actions_file" || true)"
fi

if [[ "$current_content" == "$desired_content" ]]; then
  gok "DNF actions file already configured"
else
  tmp="$(mktemp)"
  printf "%s" "$desired_content" > "$tmp"
  if is_root; then
    install -m 0644 "$tmp" "$actions_file"
  else
    sudo install -m 0644 "$tmp" "$actions_file"
  fi
  rm -f "$tmp"
  gok "Wrote ${actions_file}"
fi

# ---------- summary ----------
gtitle "Done"

gum style --border rounded --padding "1 2" --margin "1" --foreground 255 --border-foreground 213 "$(cat <<EOF
What was done:
- Verified / and /home are Btrfs and (best-effort) mounted as /root and /home
- Installed: snapper, btrfs-assistant, libdnf5-plugin-actions, gum
- Created snapper configs: root (/), home (/home)
- Restored SELinux contexts for /.snapshots and /home/.snapshots
- Enabled snapper timeline + cleanup timers
- Disabled home timeline snapshots
- (Optional) Set root retention limits
- Added dnf5 actions hooks to create pre/post snapshots for root

Next suggested test:
1) Run:  sudo dnf upgrade --refresh
2) Check: snapper -c root list | tail
EOF
)"

if gum confirm "Run a quick test upgrade (dnf upgrade --refresh) now?"; then
  run dnf -y upgrade --refresh
  gok "Upgrade completed"
  ginfo "Recent root snapshots:"
  snapper -c root list | tail -n 10
else
  ginfo "Skipped upgrade test."
fi

gok "All set."
