#date: 2025-05-05T16:58:08Z
#url: https://api.github.com/gists/be5bb134eb8254c44f7570084d871563
#owner: https://api.github.com/users/livingstaccato

#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipelines return status of the last command to exit with non-zero status,
# or zero if all commands exit successfully.
set -o pipefail

# --- Configuration ---
VAULT_VERSION="1.19.3"
ARCH="darwin_arm64" # Apple Silicon
# INSTALL_DIR="/usr/local/bin" # Ensure this is in your PATH, may require sudo
INSTALL_DIR="${HOME}/.local/bin" # Alternative user-local path

# Persistent storage configuration
CONFIG_DIR="${HOME}/.config/vault" # Directory to store vault.hcl
DATA_DIR="${HOME}/vault-data"      # Directory for persistent data storage
CONFIG_FILE="${CONFIG_DIR}/vault.hcl"
LISTENER_ADDR="127.0.0.1:8200" # Listen only on localhost

# --- Constants ---
DOWNLOAD_URL="https://releases.hashicorp.com/vault/${VAULT_VERSION}/vault_${VAULT_VERSION}_${ARCH}.zip"
CHECKSUM_URL="https://releases.hashicorp.com/vault/${VAULT_VERSION}/vault_${VAULT_VERSION}_SHA256SUMS"
ZIP_FILE="vault_${VAULT_VERSION}_${ARCH}.zip"
CHECKSUM_FILE="vault_${VAULT_VERSION}_SHA256SUMS"
EXPECTED_BINARY="vault"

# --- Helper Functions ---
check_command() {
  if ! command -v "$1" &>/dev/null; then
    echo "Error: Required command '$1' not found. Please install it." >&2
    exit 1
  fi
}

cleanup() {
  # This function is called on EXIT, cleanup temporary files
  if [[ -n "${TMP_DIR:-}" && -d "${TMP_DIR}" ]]; then
    echo "Cleaning up temporary directory: ${TMP_DIR}"
    rm -rf "${TMP_DIR}"
  fi
}

# --- Main Script Logic ---

# Register cleanup function to run on script exit
trap cleanup EXIT INT TERM

# Check dependencies
echo "Checking for required tools (curl, unzip, shasum)..."
check_command curl
check_command unzip
check_command shasum

# Create a temporary directory for downloads
TMP_DIR=$(mktemp -d)
echo "Using temporary directory: ${TMP_DIR}"
cd "${TMP_DIR}"

# --- Download and Install Vault (Similar to previous script) ---
echo "Downloading Vault v${VAULT_VERSION} (${ARCH})..."
curl --fail --location --progress-bar --output "${ZIP_FILE}" "${DOWNLOAD_URL}"

echo "Downloading SHA256 checksums..."
curl --fail --location --progress-bar --output "${CHECKSUM_FILE}" "${CHECKSUM_URL}"

echo "Verifying checksum..."
if grep "${ZIP_FILE}" "${CHECKSUM_FILE}" | shasum -a 256 -c -; then
  echo "Checksum verified successfully."
else
  echo "Error: Checksum verification failed!" >&2
  exit 1
fi

echo "Extracting Vault binary..."
unzip -o "${ZIP_FILE}" "${EXPECTED_BINARY}"
chmod +x "${EXPECTED_BINARY}"

echo "Installing Vault binary to ${INSTALL_DIR}..."
if [[ -w "${INSTALL_DIR}" && ! -d "${INSTALL_DIR}/${EXPECTED_BINARY}" ]]; then
  mv "${EXPECTED_BINARY}" "${INSTALL_DIR}/"
else
  echo "Moving binary may require administrator privileges (sudo)."
  sudo mv "${EXPECTED_BINARY}" "${INSTALL_DIR}/"
fi

INSTALLED_PATH="${INSTALL_DIR}/${EXPECTED_BINARY}"
if [[ -f "${INSTALLED_PATH}" ]] && command -v vault &>/dev/null; then
  echo "Vault installed successfully to ${INSTALLED_PATH}"
  echo "Version check:"
  "${INSTALLED_PATH}" --version
else
  echo "Error: Vault installation failed or not found in PATH." >&2
  echo "Please ensure '${INSTALL_DIR}' is in your system's PATH environment variable." >&2
  exit 1
fi

# Go back to original directory before cleanup trap runs
cd - >/dev/null

# --- Create Configuration and Data Directories ---
echo "Creating configuration directory: ${CONFIG_DIR}"
mkdir -p "${CONFIG_DIR}"

echo "Creating data directory: ${DATA_DIR}"
mkdir -p "${DATA_DIR}" # Vault process needs write access here

# --- Create Vault Configuration File (vault.hcl) ---
echo "Creating Vault configuration file: ${CONFIG_FILE}"
# Use a Heredoc to write the config file
cat <<EOF >"${CONFIG_FILE}"
# Vault Configuration File for Persistent Single-Node (Local Dev)

# Storage Backend: Use local file system
storage "file" {
  path = "${DATA_DIR}"
}

# Listener: Listen on localhost only, disable TLS for simplicity (INSECURE FOR PRODUCTION)
listener "tcp" {
  address     = "${LISTENER_ADDR}"
  tls_disable = true # !! WARNING: DO NOT USE IN PRODUCTION or NON-LOCAL scenarios !!
}

# API Address: Explicitly set for clarity
api_addr = "http://${LISTENER_ADDR}"

# Disable Memory Locking: Often needed on macOS/Docker/dev environments
disable_mlock = true

# Enable Web UI (optional but helpful)
ui = true
EOF

echo "Vault configuration written to ${CONFIG_FILE}"
echo ""
echo "-----------------------------------------------------------------------"
echo " Installation and Configuration Complete!"
echo " Vault is configured for persistent storage at: ${DATA_DIR}"
echo " Configuration file: ${CONFIG_FILE}"
echo ""
echo " NEXT STEPS (Manual):"
echo ""
echo " 1. START the Vault Server:"
echo "    Open a NEW terminal window/tab and run:"
echo "    vault server -config=\"${CONFIG_FILE}\""
echo "    (Leave this terminal running for the Vault server)"
echo ""
echo " 2. INITIALIZE & UNSEAL (Only needs init ONCE EVER):"
echo "    Open a SECOND NEW terminal window/tab."
echo "    Set the VAULT_ADDR:"
echo "    export VAULT_ADDR=\"http://${LISTENER_ADDR}\""
echo ""
echo "    If this is the *very first time* starting Vault with this config:"
echo "      Run: vault operator init"
echo "      !! CRITICAL: "**********"
echo "      !!             You WILL NEED these to unseal/login later.         !!"
echo ""
echo "    To Unseal Vault (needed every time Vault starts):"
echo "      Run 'vault status' to check if sealed."
echo "      Run 'vault operator unseal' and enter one of the Unseal Keys when prompted."
echo "      Repeat 'vault operator unseal' (usually 3 times total) until Vault is unsealed."
echo "      (Use the keys you saved from 'vault operator init')"
echo ""
echo " 3. LOGIN:"
echo  "**********"    Once unsealed, log in using the Root Token saved from 'init': "**********"
echo "    vault login <your-initial-root-token>"
echo "    (Keep this second terminal open for running Vault commands)"
echo ""
echo " 4. USE with Terraform:"
echo "    In the SECOND terminal (where you are logged in), ensure VAULT_ADDR is set:"
echo "    export VAULT_ADDR=\"http://${LISTENER_ADDR}\""
echo  "**********"    Terraform also needs a token. You can export the root token (easier for local dev): "**********"
echo "    export VAULT_TOKEN= "**********"
echo "    Or generate a more limited token if desired after logging in."
echo "    Now you can run Terraform commands in this terminal."
echo "-----------------------------------------------------------------------"

exit 0
----"

exit 0
