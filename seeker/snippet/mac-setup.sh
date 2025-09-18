#date: 2025-09-18T16:56:19Z
#url: https://api.github.com/gists/b4dff21fe54f84fee12d9a47637d93de
#owner: https://api.github.com/users/develpudu

#!/bin/bash

# Exit on error
set -e

echo "Starting macOS setup..."

# Disable Homebrew cleanup
export HOMEBREW_NO_INSTALL_CLEANUP=1

### 1. Check if Homebrew is installed, and install or update it
if command -v brew &>/dev/null; then
    echo "Homebrew is already installed, updating..."
    brew update
else
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

### 2. Install Applications
echo "Installing apps: Chromium, VSCodium, iTerm2, Zsh, Docker, Insomnia..."

brew install --cask vscodium
brew install --cask goland
brew install --cask iterm2
brew install --cask openlens
brew install --cask chatgpt
brew install --cask 1password
brew install --cask slack
brew install --cask microsoft-teams
brew install --cask chromium
brew install --cask whatsapp
brew install --cask windows-app
brew install --cask dbeaver-community
brew install 1password-cli
brew install git
brew install zsh
brew install go
brew install kubectl
brew install helm
brew install azure-cli

# Chromium is broken in first install, let's fix it 
xattr -cr /Applications/Chromium.app

echo "Setting up git configs"
# Set global Git username and email
git config --global user.name "Nadir Akdağ"
git config --global user.email "nadirakdag@outlook.com"

# Optional: Set default editor for Git to vim or nano (choose your preferred editor)
git config --global core.editor "nano"

# Set Git to always use a credential helper (for caching your credentials)
git config --global credential.helper osxkeychain

# Optional: Enable colored output for Git commands
git config --global color.ui auto


echo "Create a folder for GitHub repositories and personal projects"
mkdir -p ~/Projects


echo "Applying system preferences..."

echo "Removing default Dock apps..."
defaults write com.apple.dock persistent-apps -array

echo "Show hidden files in Finder"
defaults write com.apple.finder AppleShowAllFiles -bool true

echo "Show all filename extensions"
defaults write NSGlobalDomain AppleShowAllExtensions -bool true

echo "Show Path bar in Finder"
defaults write com.apple.finder ShowPathbar -bool true

echo "Show Status bar in Finder"
defaults write com.apple.finder ShowStatusBar -bool true


echo "Avoid creating .DS_Store on network or USB volumes"
defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true
defaults write com.apple.desktopservices DSDontWriteUSBStores -bool true

echo "Disable system sounds"
defaults write com.apple.systemsound "com.apple.sound.uiaudio.enabled" -int 0

echo "Enable Dark Mode"
defaults write NSGlobalDomain AppleInterfaceStyle -string "Dark"

echo "Change the default save location to iCloud"
defaults write NSGlobalDomain NSDocumentSaveNewDocumentsToCloud -bool false

echo "Disable 'Are you sure you want to open this file?' warning"
defaults write com.apple.LaunchServices LSQuarantine -bool false

# Enable Developer Tools in Safari
# defaults write com.apple.Safari IncludeDevelopMenu -bool true

echo "Change screenshot save location"
mkdir -p ~/Pictures/Screenshots
defaults write com.apple.screencapture location ~/Pictures/Screenshots


echo "Show Library folder in Finder"
chflags nohidden ~/Library

echo "Disable auto-correction in text fields"
defaults write NSGlobalDomain NSAutomaticSpellingCorrectionEnabled -bool false

echo "Make Finder open to Home directory instead of Recents"
defaults write com.apple.finder NewWindowTarget -string "PfLo"
defaults write com.apple.finder NewWindowTargetPath -string "file://${HOME}/"

echo "Show "This Mac" in Finder Sidebar"
defaults write com.apple.finder SidebarShowThisComputer -bool true

echo "Enable subpixel anti-aliasing for fonts"
defaults write NSGlobalDomain AppleFontSmoothing -int 2

echo "Show battery percentage in menu bar"
defaults write com.apple.menuextra.battery ShowPercent -string "YES"

echo "Disable Spotlight web suggestions"
defaults write com.apple.Spotlight orderedItems -array \
    '{"enabled" = 1; "name" = "APPLICATIONS";}' \
    '{"enabled" = 1; "name" = "SYSTEM_PREFS";}' \
    '{"enabled" = 1; "name" = "DOCUMENTS";}' \
    '{"enabled" = 1; "name" = "FOLDER";}'

killall SystemUIServer
killall Finder
killall Dock


echo "Install Oh My Zsh if not installed"
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    echo "Installing Oh My Zsh..."
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
fi

echo "Install zsh-autosuggestions"
git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions

echo "Install zsh-syntax-highlighting"
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting

echo "Add plugins to your .zshrc file"
echo "plugins=(git zsh-autosuggestions zsh-syntax-highlighting)" >> ~/.zshrc

# Change theme (optional: change to 'agnoster', 'robbyrussell', etc.)
sed -i '' 's/ZSH_THEME="robbyrussell"/ZSH_THEME="agnoster"/' ~/.zshrc

echo "Set Zsh as default shell"
chsh -s /bin/zsh

echo "set the Go workspace (e.g., for development)"
echo "export GOPATH=\$HOME/go" >> ~/.zshrc
echo "export PATH=\$GOPATH/bin:\$PATH" >> ~/.zshrc

echo "macOS setup completed! 🎉"

source ~/.zshrc