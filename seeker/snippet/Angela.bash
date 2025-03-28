#date: 2025-03-28T16:56:19Z
#url: https://api.github.com/gists/dc96e853e6ac804e82a068d38fb7dbb1
#owner: https://api.github.com/users/bradleyfrank

#!/usr/bin/env bash

HOMEBREW_INSTALL_SCRIPT="https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"
NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL "$HOMEBREW_INSTALL_SCRIPT")"

export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

cat << EOF > Brewfile
cask "1password"
cask "backblaze"
cask "calibre"
cask "fantastical"
cask "iina"
cask "keepassxc"
cask "mullvadvpn"
cask "netnewswire"
cask "nextcloud"
cask "notesnook"
cask "numi"
cask "pearcleaner"
cask "plexamp"
cask "signal"
cask "sony-ps-remote-play"
cask "spotify"
cask "tor-browser"
cask "vivaldi"
cask "vlc"
cask "waterfox"
cask "zoom"
EOF

brew bundle install
