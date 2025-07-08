#date: 2025-07-08T16:36:45Z
#url: https://api.github.com/gists/d463af779d1f591703fd6200a49eaa9c
#owner: https://api.github.com/users/swyth-dev

#!/bin/bash

  

installCursor() {

if ! [ -f /opt/cursor.appimage ]; then

echo "Installing Cursor AI IDE..."

  

# Check for sudo privileges

if [ "$(id -u)" -ne 0 ]; then

echo "This script needs sudo privileges to install Cursor AI."

echo "Please run with: sudo $0"

return 1

fi

  

# URLs for Cursor AppImage and Icon

CURSOR_URL="https://downloads.cursor.com/production/faa03b17cce93e8a80b7d62d57f5eda6bb6ab9fa/linux/x64/Cursor-1.2.2-x86_64.AppImage"

ICON_URL="https://www.cursor.com/apple-touch-icon.png"

  

# Paths for installation

APPIMAGE_PATH="/opt/cursor.appimage"

ICON_PATH="/opt/cursor.png"

DESKTOP_ENTRY_PATH="/usr/share/applications/cursor.desktop"

  

# Install curl if not installed

if ! command -v curl &> /dev/null; then

echo "curl is not installed. Installing..."

apt-get update

apt-get install -y curl

fi

  

# Download Cursor AppImage

echo "Downloading Cursor AppImage..."

if ! curl -L $CURSOR_URL -o $APPIMAGE_PATH; then

echo "Failed to download Cursor AppImage. Please check your internet connection."

return 1

fi

chmod +x $APPIMAGE_PATH

  

# Download Cursor icon

echo "Downloading Cursor icon..."

if ! curl -L $ICON_URL -o $ICON_PATH; then

echo "Failed to download Cursor icon, but continuing installation."

fi

  

# Create a .desktop entry for Cursor

echo "Creating .desktop entry for Cursor..."

cat > $DESKTOP_ENTRY_PATH <<EOL

[Desktop Entry]

Name=Cursor AI IDE

Exec=$APPIMAGE_PATH --no-sandbox

Icon=$ICON_PATH

Type=Application

Categories=Development;

EOL

  

echo "Adding cursor alias to .bashrc..."

if [ -f "$HOME/.bashrc" ]; then

cat >> $HOME/.bashrc <<EOL

  

# Cursor alias

function cursor() {

/opt/cursor.appimage --no-sandbox "\${@}" > /dev/null 2>&1 & disown

}

EOL

echo "Alias added. To use it in this terminal session, run: source $HOME/.bashrc"

else

echo "Could not find .bashrc file. Cursor can still be launched from the application menu."

fi

  

echo "Cursor AI IDE installation complete. You can find it in your application menu."

else

echo "Cursor AI IDE is already installed."

fi

}

  

installCursor