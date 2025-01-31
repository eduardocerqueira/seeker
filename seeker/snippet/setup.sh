#date: 2025-01-31T16:53:35Z
#url: https://api.github.com/gists/22587d78072c6b9cfd25c0e3d4ccae5e
#owner: https://api.github.com/users/jonasSOPAT

# get SYSLOGIC_L4TJP6.1.tar.gz https://www.syslogic.com/jetson-agx-orin/rugged-edge-ai-computer-rpc-rsl-a4agx
sudo add-apt-repository universe
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install sdkmanager


sudo apt upgrade -y
tar -xvzf SYSLOGIC_L4TJP6.1.tar.gz
cd ./SYSLOGIC_L4TJP6.1
sudo ./get_sources.bash
