#date: 2022-01-06T17:13:42Z
#url: https://api.github.com/gists/ec794ee2fb003b4c9977f9704d3cd8eb
#owner: https://api.github.com/users/Ninjeneer

# This bash script makes a fresh install of my Linux System
# Because I'm tired of breaking my distro and reinstalling everything :D 

# Update and upgrade the system
sudo apt update && sudo apt upgrade -y

# Install packages
sudo apt install terminator git tig zsh flameshot docker.io vim fonts-powerline -y

# Install Oh My Zsh
yes | sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"

# Set the Agnoster theme
sed -i 's/robbyrussell/agnoster/g' .zshrc

# Install NVM
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
echo export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")" >> .zshrc
echo [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" >> .zshrc

# Install Python 3 'pip' module
sudo apt install python3-pip -y

# Create directories
mkdir dev
mkdir softwares