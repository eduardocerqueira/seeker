#date: 2025-06-19T16:41:11Z
#url: https://api.github.com/gists/857e77c72f8d5d633222f0ba76b38ae0
#owner: https://api.github.com/users/LeviSnoot

#!/usr/bin/env bash

set -e

LOCALE_FILE="en_SE"
LOCALE_NAME="en_SE.UTF-8"
INSTALL_DIR="/usr/share/i18n/locales"
CHARMAP="UTF-8"

echo "Copying $LOCALE_FILE to $INSTALL_DIR (requires root)..."
sudo cp "$LOCALE_FILE" "$INSTALL_DIR/"

echo "Generating locale $LOCALE_NAME (requires root)..."
sudo localedef -i "$LOCALE_FILE" -f "$CHARMAP" "$LOCALE_NAME"

if locale -a | grep -q "^en_SE.utf8$"; then
    echo "Locale $LOCALE_NAME successfully installed!"
else
    echo "ERROR: Locale $LOCALE_NAME was not installed correctly."
    exit 1
fi

# Credit https://gist.github.com/bmaupin
# Locale install for Ubuntu/Debian
if [ -d /var/lib/locales/supported.d ]; then
    echo "Registering locale in /var/lib/locales/supported.d/local (Debian/Ubuntu)..."
    echo "$LOCALE_NAME UTF-8" | sudo tee /var/lib/locales/supported.d/local
    echo "Running locale-gen..."
    sudo locale-gen
fi

cat <<EOF

To use this locale for your current shell session:
  export LANG=$LOCALE_NAME

To make it your default locale, add the above line to your ~/.profile or ~/.bashrc.

To set system-wide (requires root):
  sudo localectl set-locale LANG=$LOCALE_NAME

EOF