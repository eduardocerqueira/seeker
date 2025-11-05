#date: 2025-11-05T17:14:35Z
#url: https://api.github.com/gists/cadd98813db6860c4e91c65715b72349
#owner: https://api.github.com/users/robertsinfosec

#!/usr/bin/env bash
set -euo pipefail

# =========================
# service-generator.sh
# -------------------------
# Create a systemd-managed docker-compose service skeleton:
# - Unprivileged user: service-runner (home: /opt/${ServiceName})
# - Adds service-runner to docker group
# - Systemd unit: /etc/systemd/system/${ServiceName}.service
# - Sudoers drop-in for limited service control + reboot/shutdown
# Idempotent; use --force to re-apply/overwrite artifacts.
# =========================

# -------- Colors & prefixes --------
NC="\033[0m"
BOLD="\033[1m"
C_INFO="\033[1;36m"     # cyan
C_OK="\033[1;32m"       # green
C_WARN="\033[1;33m"     # yellow
C_ERR="\033[1;31m"      # red

p_info()  { echo -e "${C_INFO}[*]${NC} $*"; }
p_ok()    { echo -e "${C_OK}[+]${NC} $*"; }
p_warn()  { echo -e "${C_WARN}[!]${NC} $*"; }
p_err()   { echo -e "${C_ERR}[-]${NC} $*" >&2; }

# -------- Defaults --------
SERVICE_USER="service-runner"
SERVICE_GROUP="$SERVICE_USER"
DOCKER_GROUP="docker"
FORCE=false
NONINTERACTIVE=false

SERVICE_NAME=""
FRIENDLY_NAME=""
DESCRIPTION=""

# -------- Helpers --------
need_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    p_err "This script must be run as root (or via sudo)."
    exit 1
  fi
}

cmd_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Normalize friendly alias for systemd Alias= (spaces -> dashes, lowercased)
alias_from_friendly() {
  local s="$1"
  s="${s// /-}"
  echo "${s,,}"
}

# -------- Arg parsing --------
print_usage() {
  cat <<EOF
Usage:
  $0 [--service-name NAME] [--friendly-name "Readable Name"] [--description "Text"] [--force] [--non-interactive]

Examples:
  $0 --service-name nextcloud \\
     --friendly-name "NextCloud Cloud Storage Service" \\
     --description "Service for controlling this local instance of NextCloud."

Flags:
  --service-name       Systemd unit name (no .service suffix); also directory under /opt.
  --friendly-name      Human-readable name used for systemd Alias and metadata.
  --description        Systemd Description= line.
  --force              Re-apply/overwrite artifacts if they already exist.
  --non-interactive    Fail if required fields are missing instead of prompting.
  -h, --help           Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service-name)     SERVICE_NAME="${2:-}"; shift 2 ;;
    --friendly-name)    FRIENDLY_NAME="${2:-}"; shift 2 ;;
    --description)      DESCRIPTION="${2:-}"; shift 2 ;;
    --force)            FORCE=true; shift ;;
    --non-interactive)  NONINTERACTIVE=true; shift ;;
    -h|--help)          print_usage; exit 0 ;;
    *) p_err "Unknown argument: $1"; print_usage; exit 2 ;;
  esac
done

# -------- Prompt if needed --------
prompt_if_empty() {
  local var_name="$1"
  local prompt="$2"
  local default="${3:-}"
  local val="${!var_name:-}"

  if [[ -z "$val" ]]; then
    if $NONINTERACTIVE; then
      p_err "Missing required: $var_name"
      exit 2
    fi
    read -r -p "$prompt${default:+ [$default]}: " input || true
    if [[ -z "$input" && -n "$default" ]]; then
      input="$default"
    fi
    eval "$var_name=\"\$input\""
  fi
}

prompt_if_empty SERVICE_NAME  "Enter service name (no spaces, e.g. nextcloud)"
prompt_if_empty FRIENDLY_NAME "Enter friendly name"       "$SERVICE_NAME service"
prompt_if_empty DESCRIPTION   "Enter description"         "Managed docker-compose service for $SERVICE_NAME"

UNIT_NAME="${SERVICE_NAME}.service"
SERVICE_HOME="/opt/${SERVICE_NAME}"
SYSTEMD_UNIT="/etc/systemd/system/${UNIT_NAME}"
SUDOERS_DROP="/etc/sudoers.d/${SERVICE_NAME}-${SERVICE_USER}"
ALIAS_NAME="$(alias_from_friendly "$FRIENDLY_NAME")"
ALIAS_UNIT="${ALIAS_NAME}.service"

# -------- Preconditions --------
need_root

p_info "Validating environment..."
if ! cmd_exists docker; then
  p_err "docker is not installed or not in PATH."
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  p_err "Docker Compose V2 not available via 'docker compose'."
  exit 1
fi
if ! cmd_exists systemctl; then
  p_err "systemctl not found."
  exit 1
fi
p_ok "Environment OK."

# -------- Ensure docker group exists --------
p_info "Ensuring group '${DOCKER_GROUP}' exists..."
if getent group "${DOCKER_GROUP}" >/dev/null; then
  p_warn "Group '${DOCKER_GROUP}' already exists."
else
  groupadd "${DOCKER_GROUP}"
  p_ok "Created group '${DOCKER_GROUP}'."
fi

# -------- Ensure service user exists and is configured --------
p_info "Ensuring user '${SERVICE_USER}' exists with home ${SERVICE_HOME}..."
if id -u "${SERVICE_USER}" >/dev/null 2>&1; then
  p_warn "User '${SERVICE_USER}' already exists."
  # Reconcile home dir if needed (idempotent & safe under --force or not)
  CURRENT_HOME="$(getent passwd "${SERVICE_USER}" | cut -d: -f6)"
  if [[ "${CURRENT_HOME}" != "${SERVICE_HOME}" ]]; then
    if $FORCE; then
      p_info "Updating home dir for '${SERVICE_USER}' from ${CURRENT_HOME} to ${SERVICE_HOME}..."
      usermod -d "${SERVICE_HOME}" "${SERVICE_USER}"
      p_ok "Home updated."
    else
      p_warn "Home is '${CURRENT_HOME}', expected '${SERVICE_HOME}'. Use --force to fix."
    fi
  fi
else
  useradd -r -m -d "${SERVICE_HOME}" -s /usr/sbin/nologin -U "${SERVICE_USER}"
  p_ok "Created user '${SERVICE_USER}' with home '${SERVICE_HOME}'."
fi

# Ensure home directory exists and ownership is correct
p_info "Ensuring ${SERVICE_HOME} exists and ownership ${SERVICE_USER}:${SERVICE_GROUP}..."
mkdir -p "${SERVICE_HOME}"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${SERVICE_HOME}"
p_ok "Home directory ready."

# -------- Add user to docker group --------
p_info "Ensuring '${SERVICE_USER}' is in group '${DOCKER_GROUP}'..."
if id -nG "${SERVICE_USER}" | tr ' ' '\n' | grep -qx "${DOCKER_GROUP}"; then
  p_warn "'${SERVICE_USER}' already in '${DOCKER_GROUP}'."
else
  usermod -aG "${DOCKER_GROUP}" "${SERVICE_USER}"
  p_ok "Added '${SERVICE_USER}' to '${DOCKER_GROUP}'."
fi

# -------- Compose project scaffolding (optional but helpful) --------
# Create a placeholder docker-compose.yml if none exists
if [[ ! -f "${SERVICE_HOME}/docker-compose.yml" ]]; then
  p_info "Creating placeholder docker-compose.yml (edit before starting the service)..."
  cat > "${SERVICE_HOME}/docker-compose.yml" <<'YML'
version: "3.9"
services:
  app:
    image: hello-world
    restart: unless-stopped
YML
  chown "${SERVICE_USER}:${SERVICE_GROUP}" "${SERVICE_HOME}/docker-compose.yml"
  p_ok "Created ${SERVICE_HOME}/docker-compose.yml."
else
  p_warn "docker-compose.yml already exists; leaving as-is."
fi

# -------- Systemd unit file --------
generate_unit() {
  p_info "Writing systemd unit: ${SYSTEMD_UNIT}"
  cat > "${SYSTEMD_UNIT}" <<UNIT
[Unit]
Description=${DESCRIPTION}
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${SERVICE_HOME}
User=${SERVICE_USER}
Group=${DOCKER_GROUP}
Environment=COMPOSE_PROJECT_NAME=${SERVICE_NAME}
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

# Security hardening (tune if needed)
ProtectSystem=full
ProtectHome=true
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
Alias=${ALIAS_UNIT}
UNIT
  p_ok "Unit file written."
}

p_info "Ensuring systemd unit ${UNIT_NAME}..."
if [[ -f "${SYSTEMD_UNIT}" ]]; then
  if $FORCE; then
    p_info "Overwriting existing unit (because --force)..."
    generate_unit
  else
    p_warn "Unit already exists; use --force to overwrite."
  fi
else
  generate_unit
fi

# -------- Sudoers drop-in for limited control --------
# Allow service-runner to:
#   systemctl start|stop|restart|enable|disable ${UNIT_NAME}
#   reboot now
#   shutdown -h now
generate_sudoers() {
  p_info "Writing sudoers drop-in: ${SUDOERS_DROP}"
  umask 077
  cat > "${SUDOERS_DROP}" <<SUDO
# Auto-generated by service-generator.sh for ${SERVICE_NAME}
User_Alias SRVUSR = ${SERVICE_USER}
Cmnd_Alias SRVCTL = /bin/systemctl start ${UNIT_NAME}, \\
                    /bin/systemctl stop ${UNIT_NAME}, \\
                    /bin/systemctl restart ${UNIT_NAME}, \\
                    /bin/systemctl enable ${UNIT_NAME}, \\
                    /bin/systemctl disable ${UNIT_NAME}
Cmnd_Alias PWRCTL = /sbin/reboot now, /sbin/shutdown -h now

SRVUSR ALL=(root) NOPASSWD: SRVCTL, PWRCTL
SUDO
  chmod 0440 "${SUDOERS_DROP}"
  p_ok "Sudoers drop-in written."
}

p_info "Ensuring sudoers drop-in for ${SERVICE_USER} and ${SERVICE_NAME}..."
if [[ -f "${SUDOERS_DROP}" ]]; then
  if $FORCE; then
    p_info "Overwriting sudoers drop-in (because --force)..."
    generate_sudoers
  else
    p_warn "Sudoers drop-in already exists; use --force to overwrite."
  fi
else
  generate_sudoers
fi

# -------- systemd reload & optional enable --------
p_info "Reloading systemd daemon..."
systemctl daemon-reload
p_ok "Systemd reloaded."

# Enable the service so Alias= is materialized and service is available at boot
p_info "Enabling ${UNIT_NAME}..."
if systemctl enable "${UNIT_NAME}" >/dev/null 2>&1; then
  p_ok "Enabled ${UNIT_NAME} (also via Alias=${ALIAS_UNIT})."
else
  p_warn "Could not enable ${UNIT_NAME}. Check unit file for issues."
fi

# -------- Summary --------
echo
echo -e "${BOLD}Summary:${NC}"
echo -e "  Service name:         ${SERVICE_NAME}"
echo -e "  Friendly name:        ${FRIENDLY_NAME}"
echo -e "  Description:          ${DESCRIPTION}"
echo -e "  Home / Working dir:   ${SERVICE_HOME}"
echo -e "  Systemd unit:         ${SYSTEMD_UNIT}"
echo -e "  Systemd alias:        ${ALIAS_UNIT}"
echo -e "  Sudoers drop-in:      ${SUDOERS_DROP}"
echo -e "  Managed user/group:   ${SERVICE_USER}:${SERVICE_GROUP} (+ ${DOCKER_GROUP})"
echo
p_ok "Done. Edit ${SERVICE_HOME}/docker-compose.yml, then:"
echo "  sudo systemctl start ${UNIT_NAME}"
echo "  sudo systemctl status ${UNIT_NAME}"
