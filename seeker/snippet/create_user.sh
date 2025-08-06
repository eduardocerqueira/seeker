#date: 2025-08-06T16:56:22Z
#url: https://api.github.com/gists/20fdcbb064b1234748cf62161625c7e1
#owner: https://api.github.com/users/chethanuk

#!/usr/bin/env bash
set -euo pipefail

# Constants
USERNAME="chethan"
GROUPNAME="data"
SHELL="/bin/bash"
SUDOERS_FILE="/etc/sudoers.d/${USERNAME}"
SSH_DIR="/home/${USERNAME}/.ssh"
AUTHORIZED_KEYS="${SSH_DIR}/authorized_keys"
PUBLIC_KEY="ssh-rsa REPLACE_WITH_YOUR_PUBLIC_KEY_HERE"
ALLOWED_CMDS=(
  "/usr/bin/su"
  "/usr/bin/apt"
  "/usr/bin/apt-get"
)

# 0. Ensure running as root
if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: Must be run as root." >&2
  exit 1
fi

# 1. Create group if missing
if ! getent group "${GROUPNAME}" >/dev/null; then
  groupadd --system "${GROUPNAME}"
  echo "Group '${GROUPNAME}' created."
else
  echo "Group '${GROUPNAME}' exists; skipping."
fi

# 2. Create user if missing
if ! id "${USERNAME}" >/dev/null 2>&1; then
  useradd \
    --create-home \
    --shell "${SHELL}" \
    --gid "${GROUPNAME}" \
    --comment "CI-managed user ${USERNAME}" \
    "${USERNAME}"
  echo "User '${USERNAME}' created."
else
  echo "User '${USERNAME}' exists; skipping."
fi

# 3. Install sudo if needed
if ! command -v sudo >/dev/null; then
  echo "Installing sudo..."
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq sudo
else
  echo "sudo is installed; skipping."
fi

# 4. Configure passwordless sudo for defined commands
echo "Configuring sudoers for '${USERNAME}'..."
# Join commands with commas, no spaces
CMD_LIST=$(IFS=, ; echo "${ALLOWED_CMDS[*]}")
cat > "${SUDOERS_FILE}" <<EOF
# CI-managed sudoers for ${USERNAME}
%${GROUPNAME} ALL=(ALL) NOPASSWD: ${CMD_LIST}
EOF
chmod 0440 "${SUDOERS_FILE}"
echo "Sudoers configured: %${GROUPNAME} ALL=(ALL) NOPASSWD: ${CMD_LIST}"

# 5. Set up SSH key login
echo "Setting up SSH directory and authorized_keys..."
mkdir -p "${SSH_DIR}"
chmod 700 "${SSH_DIR}"
echo "${PUBLIC_KEY}" > "${AUTHORIZED_KEYS}"
chmod 600 "${AUTHORIZED_KEYS}"
chown -R "${USERNAME}:${GROUPNAME}" "${SSH_DIR}"
echo "SSH key installed for '${USERNAME}'."

echo "Setup complete."
echo "User '${USERNAME}' is in group '${GROUPNAME}', can run:"
for cmd in "${ALLOWED_CMDS[@]}"; do
  echo "  • sudo ${cmd}"
done
echo "And may SSH in using the provided public key."
