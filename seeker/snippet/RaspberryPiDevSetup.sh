#date: 2021-11-17T16:54:25Z
#url: https://api.github.com/gists/559e06441b41003050e020eb74e05b35
#owner: https://api.github.com/users/l0u1sg

## Getting image and Pi up and running
# Go to https://www.raspberrypi.org/downloads/raspbian/ and download an image, then unzip it
# Download https://www.balena.io/etcher/ and install it
# Plug in microsd card, Unzip the downloaded image and use Etcher to write the downloaded image to the SD card
# Edit boot/config.txt, and uncomment hdmi_force_hotplug=1 and hdmi_drive=2
# Plug the PI in, and go through to setup prompts
# Run the following commands in terminal

## Config stuff
sudo raspi-config
  # Go to Update

## Update & upgrade packages, install new firmware
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade

## Install things needed for zsh and install zsh, oh my zsh, powerlevel10k theme, and powerline fonts
sudo apt-get install git zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
git clone https://github.com/romkatv/powerlevel10k.git $ZSH_CUSTOM/themes/powerlevel10k
nano .zshrc
  # Edit the line with ZSH_THEME to instead be (uncomment):
  # ZSH_THEME="powerlevel10k/powerlevel10k"
  # Add the following to the bottom (uncomment):
  # POWERLEVEL9K_LEFT_PROMPT_ELEMENTS=(dir vcs)
sudo apt-get install fonts-powerline
  # Go to Terminal -> Preferences, and set font to Powerline
exec zsh

## Install nvm, node, and npm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.34.0/install.sh | bash
nano .zshrc
  # Add the following to the bottom (uncomment):
  # export NVM_DIR="$HOME/.nvm"
  # [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
  # [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
exec zsh
nvm install --lts
nvm use --lts
npm install -g npm@latest

## Install other useful things
# Docker CE
# Verification that Docker is not already installed 
sudo apt-get remove docker docker-engine docker.io containerd runc
# Setup repo
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
# Install Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io