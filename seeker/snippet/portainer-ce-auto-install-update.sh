#date: 2025-10-31T16:49:03Z
#url: https://api.github.com/gists/a8b61ab7ac05f04c5883cb8179205e77
#owner: https://api.github.com/users/supermarsx

#!/usr/bin/env bash
# =============================================================================
# Portainer Auto-Installer & Auto-Updater
# =============================================================================
# Purpose
#   Idempotently install or update Portainer CE to the latest image and
#   automatically keep it up-to-date via cron. The script only recreates the
#   container when a newer image is available and verifies successful startup.
#
# What it does
#   • Validates root and Docker daemon availability
#   • Pulls the latest image `portainer/portainer-ce:latest`
#   • Compares the currently running image vs. the newly pulled image
#   • Stops Portainer gracefully (with a fallback kill if needed)
#   • Recreates the container with persistent volume `portainer_data`
#   • Waits for Portainer to be healthy (container running + HTTPS API responsive)
#   • Self‑installs to /usr/local/sbin and auto‑installs a cron.d job
#   • Emits clear logs and exits non‑zero on failure
#
# Usage
#   sudo bash portainer-auto-update.sh
#   (Runs once, installs itself to /usr/local/sbin and configures cron to run
#    daily at the configured time.)
#
# Notes
#   • Safe to re-run anytime.
#   • Uses curl with -k against 9443 because Portainer’s TLS is self‑signed.
#   • Requires a working Docker installation and network access to Docker Hub.
# =============================================================================

set -euo pipefail

# -------------------------------- CONFIG -------------------------------------
PORTAINER_IMAGE="portainer/portainer-ce:latest"     # image to deploy
CONTAINER_NAME="portainer"                          # container name
DATA_VOLUME="portainer_data"                        # named volume for /data
# Where the script will live (so cron can call it)
INSTALL_PATH="/usr/local/sbin/portainer-auto-update.sh"
# Cron schedule: every day at 03:30 (change if you like)
CRON_SCHEDULE="30 3 * * *"
CRON_FILE="/etc/cron.d/portainer-auto-update"
CRON_LOG="/var/log/portainer-auto-update.log"
# Stop timeout (seconds) before force-killing the container
STOP_TIMEOUT=30
# How long to wait (seconds) for the container to be RUNNING
WAIT_RUNNING_TIMEOUT=60
# How long to wait (seconds) for HTTPS API to answer
WAIT_HTTP_TIMEOUT=90
# Auto-cron behavior (set to 0 to skip cron install/update by default)
AUTO_CRON=1
# ----------------------------------------------------------------------------

# Pretty print helpers --------------------------------------------------------
info()  { printf "[INFO]  %s
"  "$*"; }
success(){ printf "[ OK ]  %s
"  "$*"; }
warn()  { printf "[WARN]  %s
"  "$*" >&2; }
error() { printf "[ERROR] %s
"  "$*" >&2; }

require_root() {
  # Ensure we have the privileges needed to manage Docker, write cron, etc.
  if [[ $(id -u) -ne 0 ]]; then
    error "This script must be run as root (try: sudo)."
    exit 1
  fi
}

check_docker() {
  # Verify docker CLI exists and the daemon is reachable.
  if ! command -v docker >/dev/null 2>&1; then
    error "Docker is not installed. Install Docker first."
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    error "Docker daemon is not running or not accessible."
    exit 1
  fi
}

ensure_volume() {
  # Create persistent volume if missing.
  if ! docker volume inspect "$DATA_VOLUME" >/dev/null 2>&1; then
    info "Creating volume '$DATA_VOLUME'..."
    docker volume create "$DATA_VOLUME" >/dev/null
    success "Volume '$DATA_VOLUME' ready."
  fi
}

get_current_container_image_id() {
  # Returns the image ID of the currently *configured* container (if present).
  docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Image}}' | while read -r img; do
    docker image inspect "$img" --format '{{.Id}}' 2>/dev/null || true
  done | head -n1
}

pull_latest_image() {
  info "Pulling latest image: $PORTAINER_IMAGE"
  docker pull "$PORTAINER_IMAGE" >/dev/null
  docker image inspect "$PORTAINER_IMAGE" --format '{{.Id}}'
}

stop_container_gracefully() {
  # Stop the container with a grace period; force kill if it refuses to stop.
  if docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    info "Stopping container '$CONTAINER_NAME' (timeout=${STOP_TIMEOUT}s)..."
    if ! docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1; then
      warn "Graceful stop failed; forcing kill..."
      docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    success "Container '$CONTAINER_NAME' stopped."
  fi
}

remove_container_if_exists() {
  if docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    info "Removing container '$CONTAINER_NAME'..."
    docker rm "$CONTAINER_NAME" >/dev/null
    success "Container '$CONTAINER_NAME' removed."
  fi
}

run_portainer() {
  info "Launching Portainer..."
  docker run -d \
    -p 8000:8000 \
    -p 9443:9443 \
    --name="$CONTAINER_NAME" \
    --restart=always \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$DATA_VOLUME":/data \
    "$PORTAINER_IMAGE" >/dev/null
  success "Container created."
}

wait_for_running() {
  # Poll until the container's State.Running is true or timeout is reached.
  local elapsed=0
  while (( elapsed < WAIT_RUNNING_TIMEOUT )); do
    local state
    state=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)
    if [[ "$state" == "true" ]]; then
      success "Container state is RUNNING."
      return 0
    fi
    sleep 2
    ((elapsed+=2))
  done
  error "Container did not reach RUNNING state within ${WAIT_RUNNING_TIMEOUT}s."
  return 1
}

wait_for_https_api() {
  # Try hitting Portainer API over https://127.0.0.1:9443/api/status
  # Use -k to ignore self-signed cert; treat any HTTP success as OK.
  local elapsed=0
  while (( elapsed < WAIT_HTTP_TIMEOUT )); do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS -k https://127.0.0.1:9443/api/status >/dev/null 2>&1; then
        success "HTTPS API responded on 9443."
        return 0
      fi
    else
      warn "curl not found; skipping HTTPS readiness probe."
      return 0
    fi
    sleep 3
    ((elapsed+=3))
  done
  error "Portainer HTTPS API did not respond within ${WAIT_HTTP_TIMEOUT}s."
  return 1
}

recreate_if_needed() {
  # Pull, compare, and (re)create container only when needed.
  info "Checking current Portainer container/image..."
  local current_id latest_id
  current_id=$(get_current_container_image_id || true)
  latest_id=$(pull_latest_image)

  if [[ -n "$current_id" && "$current_id" == "$latest_id" ]]; then
    info "Already on latest image; nothing to do."
    return 0
  fi

  info "New image detected or container missing. Proceeding to update..."
  stop_container_gracefully
  remove_container_if_exists
  run_portainer
  wait_for_running
  wait_for_https_api
}

self_install() {
  # Ensure the script lives at $INSTALL_PATH with executable perms for cron.
  if [[ "$0" != "$INSTALL_PATH" ]]; then
    info "Installing script to $INSTALL_PATH ..."
    cp "$0" "$INSTALL_PATH"
    chmod 0755 "$INSTALL_PATH"
    success "Installed at $INSTALL_PATH"
  else
    # Ensure it has sane perms even if already there
    chmod 0755 "$INSTALL_PATH" 2>/dev/null || true
  fi
}

cron_exists() {
  [[ -f "$CRON_FILE" ]]
}

reload_cron_service() {
  if command -v systemctl >/dev/null 2>&1; then
    if systemctl is-active --quiet cron 2>/dev/null; then
      systemctl reload cron 2>/dev/null || true
    elif systemctl is-active --quiet crond 2>/dev/null; then
      systemctl reload crond 2>/dev/null || true
    fi
  fi
}

desired_cron_content() {
  cat <<EOF
# Managed by portainer-auto-update.sh — do not edit by hand
$CRON_SCHEDULE root $INSTALL_PATH >>$CRON_LOG 2>&1
EOF
}

install_or_update_cron() {
  info "Ensuring cron job at $CRON_FILE matches desired state..."
  umask 022
  local tmp
  tmp=$(mktemp)
  desired_cron_content > "$tmp"

  if cron_exists; then
    if cmp -s "$tmp" "$CRON_FILE"; then
      info "Cron already up-to-date."
      rm -f "$tmp"
      return 0
    else
      info "Updating existing cron job..."
      cp "$tmp" "$CRON_FILE"
      chmod 0644 "$CRON_FILE"
      rm -f "$tmp"
      reload_cron_service
      success "Cron updated: will run as '$CRON_SCHEDULE'."
      return 0
    fi
  else
    info "Installing new cron job..."
    cp "$tmp" "$CRON_FILE"
    chmod 0644 "$CRON_FILE"
    rm -f "$tmp"
    reload_cron_service
    success "Cron installed: will run as '$CRON_SCHEDULE'."
    return 0
  fi
}

remove_cron() {
  if cron_exists; then
    info "Removing cron job at $CRON_FILE ..."
    rm -f "$CRON_FILE"
    reload_cron_service
    success "Cron job removed."
  else
    info "No cron job to remove (not found at $CRON_FILE)."
  fi
}

main() {
  # ------------------------- argument parsing -------------------------------
  # Supported flags:
  #   --no-cron               : do not install/update cron on this run
  #   --remove-cron           : remove cron job and exit
  #   --cron-schedule "..."   : override CRON_SCHEDULE for this run (and save)
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-cron)
        AUTO_CRON=0
        shift
        ;;
      --remove-cron)
        require_root
        self_install 2>/dev/null || true
        remove_cron
        exit 0
        ;;
      --cron-schedule)
        if [[ $# -lt 2 ]]; then
          error "--cron-schedule requires a value, e.g. \"15 4 * * *\""
          exit 2
        fi
        CRON_SCHEDULE="$2"
        shift 2
        ;;
      *)
        warn "Unknown argument: $1 (ignored)"
        shift
        ;;
    esac
  done

  # ----------------------------- main flow ---------------------------------
  require_root
  check_docker
  ensure_volume

  # Ensure the script lives where cron expects it
  self_install

  # Install/update cron unless explicitly disabled
  if [[ "${AUTO_CRON:-1}" -eq 1 ]]; then
    install_or_update_cron
  else
    info "AUTO_CRON disabled for this run; skipping cron install/update."
  fi

  # Perform update logic now.
  recreate_if_needed

  success "Done. Portainer is up-to-date and reachable."
}

main "$@"
