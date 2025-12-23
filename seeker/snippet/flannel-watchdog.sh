#date: 2025-12-23T17:08:00Z
#url: https://api.github.com/gists/11689c8d2c6fc5fff7dbd78ed3baae69
#owner: https://api.github.com/users/mattmattox

#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Configuration (override via environment or systemd EnvironmentFile)
###############################################################################
LOOP_DELAY="${LOOP_DELAY:-30}"          # Seconds between checks
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"     # Max wait for flannel recovery after restart
WAIT_INTERVAL="${WAIT_INTERVAL:-5}"     # Poll interval during recovery wait
DRY_RUN="${DRY_RUN:-0}"                # 1 = detect only, do not restart
MATCH_PATTERN="${MATCH_PATTERN:-flannel}" # docker ps match (name/image), case-insensitive

ANNOT_KEY="flannel.alpha.coreos.com/backend-data"

###############################################################################
# Helpers
###############################################################################
techo() {
  printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*"
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

###############################################################################
# Root check (service should run as root if it needs docker/kubeconfig access)
###############################################################################
if [[ "${EUID}" -ne 0 ]]; then
  techo "ERROR: This script must be run as root."
  exit 1
fi

###############################################################################
# Kubernetes SSL / kubeconfig detection
###############################################################################
if [ -d /opt/rke/etc/kubernetes/ssl ]; then
  K8S_SSLDIR=/opt/rke/etc/kubernetes/ssl
else
  K8S_SSLDIR=/etc/kubernetes/ssl
fi

KUBECONFIG="${K8S_SSLDIR}/kubecfg-kube-node.yaml"
NODE_NAME="$(hostname -s)"

###############################################################################
# Functions
###############################################################################
get_annotation_value() {
  local val
  val="$(
    kubectl --kubeconfig "${KUBECONFIG}" get node "${NODE_NAME}" -o \
      "jsonpath={.metadata.annotations['${ANNOT_KEY}']}" 2>/dev/null || true
  )"

  [[ "${val}" == "null" ]] && val=""
  printf '%s' "${val}" | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

find_flannel_container_ids() {
  docker ps --format '{{.ID}} {{.Names}} {{.Image}}' \
    | grep -i -- "${MATCH_PATTERN}" \
    | awk '{print $1}' \
    || true
}

wait_for_containers_running() {
  local start now elapsed state cid
  local ids=("$@")

  start="$(date +%s)"
  while true; do
    local all_running=true

    for cid in "${ids[@]}"; do
      state="$(docker inspect -f '{{.State.Status}}' "${cid}" 2>/dev/null || true)"
      if [[ "${state}" != "running" ]]; then
        all_running=false
        techo "Waiting for container ${cid} (state=${state:-unknown})..."
      fi
    done

    if [[ "${all_running}" == "true" ]]; then
      techo "Container(s) matching '${MATCH_PATTERN}' are running."
      return 0
    fi

    now="$(date +%s)"
    elapsed="$((now - start))"
    if (( elapsed >= WAIT_TIMEOUT )); then
      techo "ERROR: Timed out waiting for container(s) to be running."
      return 1
    fi

    sleep "${WAIT_INTERVAL}"
  done
}

wait_for_annotation_present() {
  local start now elapsed val
  start="$(date +%s)"

  while true; do
    val="$(get_annotation_value || true)"
    if [[ -n "${val}" ]]; then
      techo "Annotation '${ANNOT_KEY}' is populated."
      return 0
    fi

    techo "Waiting for annotation '${ANNOT_KEY}'..."
    now="$(date +%s)"
    elapsed="$((now - start))"
    if (( elapsed >= WAIT_TIMEOUT )); then
      techo "ERROR: Timed out waiting for annotation '${ANNOT_KEY}'."
      return 1
    fi

    sleep "${WAIT_INTERVAL}"
  done
}

check_and_recover() {
  local annot_val
  annot_val="$(get_annotation_value || true)"

  if [[ -z "${annot_val}" ]]; then
    if is_truthy "${DRY_RUN}"; then
      techo "DETECTED: Annotation missing/empty on node '${NODE_NAME}' (dry-run; no restart performed)."
      return 0
    fi

    techo "DETECTED: Annotation missing/empty on node '${NODE_NAME}'. Initiating recovery."

    mapfile -t ids < <(find_flannel_container_ids)
    if (( ${#ids[@]} == 0 )); then
      techo "WARNING: No containers matched pattern '${MATCH_PATTERN}' via 'docker ps'."
      return 0
    fi

    for cid in "${ids[@]}"; do
      techo "Restarting container ${cid}"
      docker restart "${cid}" >/dev/null
    done

    wait_for_containers_running "${ids[@]}"
    wait_for_annotation_present
    techo "Recovery completed."
  else
    techo "OK: Annotation present on node '${NODE_NAME}'."
  fi
}

###############################################################################
# Main loop
###############################################################################
techo "Starting watchdog (node=${NODE_NAME}, delay=${LOOP_DELAY}s, dry_run=${DRY_RUN}, match='${MATCH_PATTERN}')"

while true; do
  check_and_recover || techo "WARNING: iteration completed with errors."
  sleep "${LOOP_DELAY}"
done