#date: 2025-05-22T16:55:59Z
#url: https://api.github.com/gists/18a9599744bd6d0146f23f994b140fe2
#owner: https://api.github.com/users/szmyty

#!/usr/bin/env bash
set -euo pipefail

CERT_DIR="$(pwd)"
DOMAIN="localhost 127.0.0.1 ::1"

echo "📍 Output directory: $CERT_DIR"

# Check if mkcert is installed
if ! command -v mkcert &> /dev/null; then
  echo "❌ mkcert is not installed."

  # Recommend install instructions
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "💡 Install with: brew install mkcert && mkcert -install"
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "💡 Download from: https://github.com/FiloSottile/mkcert/releases"
  else
    echo "💡 Visit https://github.com/FiloSottile/mkcert for install instructions."
  fi
  exit 1
fi

# Ensure local CA is installed
mkcert -install

# Generate certs
echo "📦 Generating certificate for: $DOMAIN"
mkcert -cert-file "$CERT_DIR/localhost.pem" -key-file "$CERT_DIR/localhost-key.pem" $DOMAIN

echo "✅ Certificates generated:"
echo "  - $CERT_DIR/localhost.pem"
echo "  - $CERT_DIR/localhost-key.pem"
