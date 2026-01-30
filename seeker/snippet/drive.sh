#date: 2026-01-30T17:26:07Z
#url: https://api.github.com/gists/d053828228b57722c6bcb29cdb13d2e2
#owner: https://api.github.com/users/Gilbertly

#!/usr/bin/env bash
set -euo pipefail

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
info() { printf "• %s\n" "$*"; }
warn() { printf "\033[33m! %s\033[0m\n" "$*"; }
err()  { printf "\033[31m✗ %s\033[0m\n" "$*" >&2; }
ok()   { printf "\033[32m✓ %s\033[0m\n" "$*"; }

require_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    err "This installer is for macOS only."
    exit 1
  fi
}

prompt() {
  local var_name="$1"
  local message="$2"
  local secret="${3: "**********"
  local default="${4:-}"

  local value=""
  while [[ -z "$value" ]]; do
    if [[ -n "$default" ]]; then
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"[ "**********"[ "**********"  "**********"" "**********"$ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"" "**********"  "**********"= "**********"= "**********"  "**********"" "**********"t "**********"r "**********"u "**********"e "**********"" "**********"  "**********"] "**********"] "**********"; "**********"  "**********"t "**********"h "**********"e "**********"n "**********"
        read -r -s -p "$message (default: $default): " value
        echo
      else
        read -r -p "$message (default: $default): " value
      fi
      value="${value:-$default}"
    else
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"[ "**********"[ "**********"  "**********"" "**********"$ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"" "**********"  "**********"= "**********"= "**********"  "**********"" "**********"t "**********"r "**********"u "**********"e "**********"" "**********"  "**********"] "**********"] "**********"; "**********"  "**********"t "**********"h "**********"e "**********"n "**********"
        read -r -s -p "$message: " value
        echo
      else
        read -r -p "$message: " value
      fi
    fi
  done

  # shellcheck disable=SC2163
  export "$var_name=$value"
}

open_full_disk_access_settings() {
  # This opens the Privacy & Security pane. Apple doesn't provide a perfect deep-link
  # to the exact sub-page on all macOS versions, so we open the closest reliable pane.
  warn "Opening System Settings → Privacy & Security (so you can enable Full Disk Access)..."
  open "x-apple.systempreferences:com.apple.preference.security"
  echo
  bold "Manual step required (macOS security): Enable Full Disk Access"
  cat <<'TXT'
1) System Settings → Privacy & Security
2) Scroll to "Full Disk Access"
3) Enable for:
   - Terminal (or iTerm, whichever you used to run this script)
   - (Optional) rclone, if it appears
4) Quit Terminal completely and reopen it
5) Then re-run this script.

Why: macOS may block processes from using /Volumes mounts, which causes rclone to exit with:
  "open /Volumes/...: operation not permitted"
TXT
  echo
}

ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    ok "Homebrew is installed."
    return
  fi

  bold "Homebrew not found."
  read -r -p "Install Homebrew now? (y/N): " ans
  if [[ "${ans:-}" != "y" && "${ans:-}" != "Y" ]]; then
    err "Homebrew is required to install macFUSE automatically. Aborting."
    exit 1
  fi

  info "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi

  ok "Homebrew installed."
}

ensure_macfuse() {
  if [[ -d /Library/Filesystems/macfuse.fs ]]; then
    ok "macFUSE appears installed."
    return
  fi

  bold "macFUSE is required for rclone mount on macOS."
  read -r -p "Install macFUSE now (via Homebrew cask)? (y/N): " ans
  if [[ "${ans:-}" != "y" && "${ans:-}" != "Y" ]]; then
    err "macFUSE is required. Aborting."
    exit 1
  fi

  info "Installing macFUSE..."
  brew install --cask macfuse

  warn "IMPORTANT: macOS may block macFUSE until you approve it."
  warn "Go to: System Settings → Privacy & Security → Allow (macFUSE), then restart if prompted."
  read -r -p "Press Enter to continue after approving (and rebooting if required)... " _
}

ensure_rclone_official() {
  # Remove brew rclone if present to avoid mount limitations
  if command -v brew >/dev/null 2>&1; then
    if brew list --formula 2>/dev/null | grep -qx "rclone"; then
      warn "Homebrew rclone detected. Uninstalling to avoid mount limitations..."
      brew uninstall rclone || true
    fi
  fi

  info "Installing official rclone binary..."
  set +e
  curl -fsSL https://rclone.org/install.sh | sudo bash
  local rc=$?
  set -e

  # Upstream installer may return non-zero even when rclone is already installed
  if [[ $rc -ne 0 ]]; then
    warn "rclone installer returned exit code ${rc}. Continuing if rclone is present..."
  fi

  hash -r

  if ! command -v rclone >/dev/null 2>&1; then
    err "rclone not found after install attempt."
    exit 1
  fi

  ok "rclone installed at: $(command -v rclone)"
  rclone version | head -n 2 || true
}

write_rclone_config() {
  local config_dir="${HOME}/.config/rclone"
  local config_file="${config_dir}/rclone.conf"

  mkdir -p "$config_dir"
  chmod 700 "$config_dir"

  if [[ -f "$config_file" ]] && grep -q "^\[${R2_REMOTE}\]$" "$config_file"; then
    warn "An rclone remote named '${R2_REMOTE}' already exists in ${config_file}."
    read -r -p "Overwrite remote '${R2_REMOTE}' config? (y/N): " ow
    if [[ "${ow:-}" != "y" && "${ow:-}" != "Y" ]]; then
      info "Keeping existing remote config."
      return
    fi
    perl -0777 -i -pe "s/\\n?\\[\\Q${R2_REMOTE}\\E\\]\\n.*?(?=\\n\\[|\\z)//s" "$config_file"
  fi

  info "Writing rclone config for remote '${R2_REMOTE}'..."
  {
    echo "[${R2_REMOTE}]"
    echo "type = s3"
    echo "provider = Cloudflare"
    echo "access_key_id = "**********"
    echo "secret_access_key = "**********"
    echo "endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    echo "region = auto"
    echo "acl = private"
  } >> "$config_file"

  chmod 600 "$config_file"
  ok "rclone config written to ${config_file} (permissions set to 600)."
}

test_remote() {
  info "Testing access to bucket '${R2_BUCKET}'..."
  rclone lsd "${R2_REMOTE}:" >/dev/null
  ok "Remote is reachable."
  rclone lsf "${R2_REMOTE}:${R2_BUCKET}" >/dev/null
  ok "Bucket '${R2_BUCKET}' is accessible."
}

cleanup_stale_mount() {
  if mount | grep -q " on ${MOUNTPOINT} "; then
    warn "Something is already mounted at ${MOUNTPOINT}. Attempting to unmount..."
    diskutil unmount force "$MOUNTPOINT" >/dev/null 2>&1 || true
  fi
  pkill -f "rclone mount.*${MOUNTPOINT}" >/dev/null 2>&1 || true
}

reset_mountpoint_dir() {
  # Avoid deleting a live mount; ensure it's not mounted first.
  cleanup_stale_mount

  info "Resetting mountpoint directory at ${MOUNTPOINT} (requires sudo)..."
  sudo rm -rf "$MOUNTPOINT"
  sudo mkdir -p "$MOUNTPOINT"
  sudo chown "$USER":staff "$MOUNTPOINT"
  sudo chmod 755 "$MOUNTPOINT"
}

mount_daemon() {
  local log_file="$HOME/Library/Logs/r2drive.${R2_REMOTE}.${R2_BUCKET}.log"
  mkdir -p "$HOME/Library/Logs"

  info "Mounting '${R2_REMOTE}:${R2_BUCKET}' at '${MOUNTPOINT}'..."
  info "Logging to: ${log_file}"

  rclone mount "${R2_REMOTE}:${R2_BUCKET}" "$MOUNTPOINT" \
    --vfs-cache-mode full \
    --vfs-write-back 10s \
    --vfs-cache-max-size "${VFS_CACHE_MAX_SIZE}" \
    --vfs-cache-max-age "${VFS_CACHE_MAX_AGE}" \
    --dir-cache-time "${DIR_CACHE_TIME}" \
    --noappledouble \
    --noapplexattr \
    --log-level INFO \
    --log-file "$log_file" \
    --daemon

  sleep 1

  if mount | grep -q " on ${MOUNTPOINT} "; then
    ok "Mounted successfully."
    return 0
  fi

  warn "Mount did not appear active yet."
  warn "Checking log for common macOS permission issues..."

  if [[ -f "$log_file" ]] && tail -n 200 "$log_file" | grep -qi "operation not permitted"; then
    warn "Detected: operation not permitted when accessing ${MOUNTPOINT}"
    open_full_disk_access_settings
    exit 1
  fi

  warn "Check log: ${log_file}"
  warn "For direct errors, run without daemon:"
  warn "  rclone mount \"${R2_REMOTE}:${R2_BUCKET}\" \"${MOUNTPOINT}\" --vfs-cache-mode full --vfs-write-back 10s --noappledouble --noapplexattr -vv"
  exit 1
}

install_launchagent() {
  local plist_dir="${HOME}/Library/LaunchAgents"
  local safe_bucket="${R2_BUCKET// /_}"
  local plist_path="${plist_dir}/com.cloudflare.r2drive.${R2_REMOTE}.${safe_bucket}.plist"
  local rclone_path
  rclone_path="$(command -v rclone)"

  mkdir -p "$plist_dir"
  mkdir -p "$HOME/Library/Logs"

  cat > "$plist_path" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.cloudflare.r2drive.${R2_REMOTE}.${safe_bucket}</string>

    <key>ProgramArguments</key>
    <array>
      <string>${rclone_path}</string>
      <string>mount</string>
      <string>${R2_REMOTE}:${R2_BUCKET}</string>
      <string>${MOUNTPOINT}</string>

      <string>--vfs-cache-mode</string>
      <string>full</string>

      <string>--vfs-write-back</string>
      <string>10s</string>

      <string>--vfs-cache-max-size</string>
      <string>${VFS_CACHE_MAX_SIZE}</string>

      <string>--vfs-cache-max-age</string>
      <string>${VFS_CACHE_MAX_AGE}</string>

      <string>--dir-cache-time</string>
      <string>${DIR_CACHE_TIME}</string>

      <string>--noappledouble</string>
      <string>--noapplexattr</string>

      <string>--log-level</string>
      <string>INFO</string>

      <string>--log-file</string>
      <string>${HOME}/Library/Logs/r2drive.${R2_REMOTE}.${safe_bucket}.log</string>
    </array>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>${HOME}/Library/Logs/r2drive.${R2_REMOTE}.${safe_bucket}.out.log</string>
    <key>StandardErrorPath</key>
    <string>${HOME}/Library/Logs/r2drive.${R2_REMOTE}.${safe_bucket}.err.log</string>
  </dict>
</plist>
EOF

  ok "LaunchAgent created: ${plist_path}"
  launchctl unload "$plist_path" >/dev/null 2>&1 || true
  launchctl load "$plist_path"
  ok "LaunchAgent loaded. It will auto-mount on login."
}

main() {
  require_macos

  bold "Cloudflare R2 Drive Installer (macOS) — rclone + macFUSE"
  echo

  ensure_homebrew
  ensure_macfuse
  ensure_rclone_official

  echo
  bold "R2 Configuration"
  info "Create credentials in Cloudflare Dashboard → R2 → Manage R2 API Tokens."
  echo

  prompt R2_ACCOUNT_ID "Cloudflare Account ID"
  prompt R2_ACCESS_KEY_ID "R2 Access Key ID"
  prompt R2_SECRET_ACCESS_KEY "R2 Secret Access Key" true
  prompt R2_BUCKET "R2 Bucket name (e.g. mount-drive)"
  prompt R2_REMOTE "Name this rclone remote" false "r2"

  echo
  bold "Mount Options"
  prompt MOUNT_NAME "Volume name under /Volumes (shows in Finder → Go → Computer)" false "Cloudflare Drive"
  MOUNTPOINT="/Volumes/${MOUNT_NAME}"

  prompt VFS_CACHE_MAX_SIZE "VFS cache max size (e.g. 5G, 20G)" false "10G"
  prompt VFS_CACHE_MAX_AGE "VFS cache max age (e.g. 1h, 24h)" false "1h"
  prompt DIR_CACHE_TIME "Directory cache time (e.g. 5m, 1h)" false "5m"

  echo
  write_rclone_config
  test_remote

  # Make mountpoint stable + owned, every run
  reset_mountpoint_dir

  # Mount with Finder-stability flags
  mount_daemon

  echo
  read -r -p "Install auto-mount on login (LaunchAgent)? (y/N): " auto
  if [[ "${auto:-}" == "y" || "${auto:-}" == "Y" ]]; then
    install_launchagent
  else
    info "Skipping auto-mount. To mount later:"
    echo "  rclone mount ${R2_REMOTE}:${R2_BUCKET} \"${MOUNTPOINT}\" --vfs-cache-mode full --vfs-write-back 10s --noappledouble --noapplexattr --daemon"
  fi

  echo
  ok "Done!"
  info "Open Finder → Go → Computer → ${MOUNT_NAME}"
}

main "$@"