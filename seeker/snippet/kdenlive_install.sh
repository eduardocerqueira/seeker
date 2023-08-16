#date: 2023-08-16T17:03:02Z
#url: https://api.github.com/gists/255a0d65130919c5c361ae40e7c1efcb
#owner: https://api.github.com/users/techworldthink

# Add the KDEnlive stable repository
sudo add-apt-repository ppa:kdenlive/kdenlive-stable

# Install KDEnlive video editing software
sudo apt install kdenlive

# Download MediaArea repository's GPG key and add it to trusted keys
sudo wget -qO /etc/apt/trusted.gpg.d/mediaarea.asc https://mediaarea.net/repo/deb/ubuntu/pubkey.gpg

# Add MediaArea repository to the sources list
echo "deb https://mediaarea.net/repo/deb/ubuntu $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/mediaarea.list

# Update package information from repositories
sudo apt update

# Install MediaInfo tool
sudo apt install -y mediainfo

# Check installed MediaInfo version
mediainfo --version

# Install Glaxnimate using Snap package manager
snap install glaxnimate

# Check installed Glaxnimate version
glaxnimate --version

# Check installed kdenlive version
kdenlive --version
