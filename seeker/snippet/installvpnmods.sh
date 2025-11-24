#date: 2025-11-24T17:11:46Z
#url: https://api.github.com/gists/90d830a2b83ea858672ed7716cec4b50
#owner: https://api.github.com/users/5txdp5trs4-create

#!/bin/bash
# script-instalacao-integrado.sh
# Integra AdGuard Home (container) + firewalld + redirecionamento DNS -> wg0
# Mantém wg-easy e Docker presentes (preserva fluxo do seu script original).
# RODAR COMO ROOT

set -euo pipefail

LOG_FILE="/var/log/install_wg_easy_with_adguard.log"
ADGUARD_WEB_PORT=3000    # porta web do AdGuard (altere se quiser)
ADGUARD_IMAGE="adguard/adguardhome:latest"
ADGUARD_DATA_DIR="/opt/adguard"
SERVICE_NAME="wg-adguard-redirect.service"

log() { echo "[$(date '+%F %T')] $1" | tee -a "$LOG_FILE"; }

# Check root
if [[ $EUID -ne 0 ]]; then
  echo "Run as root"
  exit 1
fi

# Helper: ensure docker present (calls your install_docker if absent)
ensure_docker() {
  if ! command -v docker &>/dev/null; then
    log "Docker not found. Attempting to install Docker (this script expects your original install_docker function)."
    # Try minimal install path for Oracle/CentOS if docker not present
    if command -v yum &>/dev/null; then
      yum install -y yum-utils device-mapper-persistent-data lvm2 >>"$LOG_FILE" 2>&1 || true
      yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo >>"$LOG_FILE" 2>&1 || true
      yum install -y docker-ce --setopt=obsoletes=0 >>"$LOG_FILE" 2>&1 || true
      systemctl enable --now docker >>"$LOG_FILE" 2>&1 || true
    fi
  fi
  if ! command -v docker &>/dev/null; then
    log "Docker still not found. Aborting."
    exit 1
  fi
  log "Docker ok"
}

# Ensure functions from your original script are available:
# If you included your original script file, source it here. If not, we assume you will run original first.
# Example (uncomment if you have original script file):
# source /path/to/your/original_install_wg_easy.sh

# Create log dir
mkdir -p "$(dirname "$LOG_FILE")"

log "Starting integrated installation: AdGuard + firewalld + DNS redirect (wg0 -> AdGuard)."

# Ensure docker is present
ensure_docker

# 1) Create AdGuard data dirs (persistent)
log "Creating AdGuard data volumes at ${ADGUARD_DATA_DIR}"
mkdir -p "${ADGUARD_DATA_DIR}/work" "${ADGUARD_DATA_DIR}/conf"
chown -R 999:999 "${ADGUARD_DATA_DIR}" 2>/dev/null || true

# 2) Pull and run AdGuard container BUT do NOT bind port 53 to host.
#    We'll let Docker assign a container IP and then redirect WG DNS to that IP.
log "Pulling AdGuard image ${ADGUARD_IMAGE}..."
docker pull "${ADGUARD_IMAGE}" >>"$LOG_FILE" 2>&1

# Remove existing adguard container if exists (to recreate)
if docker ps -a --format '{{.Names}}' | grep -q "^adguardhome$"; then
  log "Removing existing adguardhome container..."
  docker stop adguardhome >>"$LOG_FILE" 2>&1 || true
  docker rm adguardhome >>"$LOG_FILE" 2>&1 || true
fi

log "Starting adguardhome container (no host DNS ports mapped)..."
docker run -d --name adguardhome \
  -v "${ADGUARD_DATA_DIR}/work":/opt/adguard/work \
  -v "${ADGUARD_DATA_DIR}/conf":/opt/adguard/conf \
  -p ${ADGUARD_WEB_PORT}:3000/tcp \
  --restart unless-stopped \
  "${ADGUARD_IMAGE}" >>"$LOG_FILE" 2>&1

# Confirm container running
sleep 2
if ! docker ps --format '{{.Names}}' | grep -q "^adguardhome$"; then
  log "AdGuard container failed to start. Check 'docker logs adguardhome'. Aborting."
  exit 1
fi

log "AdGuard container started."

# 3) Configure firewalld rules (install if needed) and open ports host needs (we do NOT open 53)
if ! command -v firewall-cmd &>/dev/null; then
  log "Installing firewalld..."
  yum install -y firewalld >>"$LOG_FILE" 2>&1 || { log "firewalld install failed"; exit 1; }
  systemctl enable --now firewalld >>"$LOG_FILE" 2>&1 || true
fi

log "Adding required firewalld rules (51820/udp, wg-easy panel port, adguard web port)."
# Keep panel port dynamic later; here assume standard 80 for wg-easy panel unless your run_wg_easy sets otherwise
# If your run_wg_easy uses PANEL_PORT exported, we'll still ensure 80 is open as default; user can change afterwards.
firewall-cmd --permanent --add-port=51820/udp >>"$LOG_FILE" 2>&1 || true
firewall-cmd --permanent --add-port=80/tcp >>"$LOG_FILE" 2>&1 || true
firewall-cmd --permanent --add-port=${ADGUARD_WEB_PORT}/tcp >>"$LOG_FILE" 2>&1 || true
firewall-cmd --reload >>"$LOG_FILE" 2>&1 || true
log "firewalld rules applied."

# 4) Run your existing wg-easy flow (assumes run_wg_easy function exists in your original script)
if type run_wg_easy &>/dev/null; then
  log "Running run_wg_easy() from original script..."
  run_wg_easy
else
  log "run_wg_easy() not found in this environment. Make sure you run the original wg-easy part before running this script or source the original script."
fi

# 5) Determine AdGuard container IP on docker network and add iptables NAT DNAT rules
log "Obtendo IP interno do container AdGuard..."
ADG_IP=""
TRIES=0
while [[ -z "$ADG_IP" && $TRIES -lt 20 ]]; do
  ADG_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' adguardhome 2>/dev/null || echo "")
  if [[ -n "$ADG_IP" ]]; then break; fi
  TRIES=$((TRIES+1))
  sleep 1
done

if [[ -z "$ADG_IP" ]]; then
  log "Falha ao obter IP do container AdGuard. Aborting."
  exit 1
fi
log "AdGuard container IP: $ADG_IP"

# Apply idempotent iptables NAT rules: redirect wg0 DNS -> AdGuard container
log "Aplicando regras iptables para redirecionamento DNS (wg0 -> ${ADG_IP}:53)."

# remove duplicates if exist
iptables -t nat -D PREROUTING -i wg0 -p udp --dport 53 -j DNAT --to-destination ${ADG_IP}:53 2>/dev/null || true
iptables -t nat -D PREROUTING -i wg0 -p tcp --dport 53 -j DNAT --to-destination ${ADG_IP}:53 2>/dev/null || true
iptables -C FORWARD -d ${ADG_IP} -j ACCEPT 2>/dev/null || true

# add
iptables -t nat -A PREROUTING -i wg0 -p udp --dport 53 -j DNAT --to-destination ${ADG_IP}:53
iptables -t nat -A PREROUTING -i wg0 -p tcp --dport 53 -j DNAT --to-destination ${ADG_IP}:53
iptables -C FORWARD -d ${ADG_IP} -j ACCEPT 2>/dev/null || iptables -A FORWARD -d ${ADG_IP} -j ACCEPT

log "Regras iptables adicionadas."

# 6) Persistência: create systemd service to reapply rules on boot (uses container IP resolution each boot)
cat > /etc/systemd/system/${SERVICE_NAME} <<EOF
[Unit]
Description=Redirect wg0 DNS to AdGuard container
After=network.target docker.service
Wants=docker.service

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'ADG_IP=\$(docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" adguardhome 2>/dev/null || echo ""); if [[ -n "\$ADG_IP" ]]; then iptables -t nat -C PREROUTING -i wg0 -p udp --dport 53 -j DNAT --to-destination \$ADG_IP:53 2>/dev/null || iptables -t nat -A PREROUTING -i wg0 -p udp --dport 53 -j DNAT --to-destination \$ADG_IP:53; iptables -t nat -C PREROUTING -i wg0 -p tcp --dport 53 -j DNAT --to-destination \$ADG_IP:53 2>/dev/null || iptables -t nat -A PREROUTING -i wg0 -p tcp --dport 53 -j DNAT --to-destination \$ADG_IP:53; iptables -C FORWARD -d \$ADG_IP -j ACCEPT 2>/dev/null || iptables -A FORWARD -d \$ADG_IP -j ACCEPT; fi'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now ${SERVICE_NAME} >>"$LOG_FILE" 2>&1 || true
log "Systemd service ${SERVICE_NAME} created and enabled."

log "Instalação integrada concluída."
log " - wg-easy painel: http://${WG_HOST:-<seu-host>}:${PANEL_PORT:-80}"
log " - AdGuard web (admin): http://${WG_HOST:-<seu-host>}:${ADGUARD_WEB_PORT}  (se quiser acesso remoto, abra esta porta na OCI)"
log "Nota: AdGuard DNS não está exposto publicamente; DNS de clientes WireGuard é forçado para AdGuard via iptables."
