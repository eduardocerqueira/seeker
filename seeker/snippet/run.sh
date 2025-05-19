#date: 2025-05-19T16:53:50Z
#url: https://api.github.com/gists/c4289b34d56406cb43c4dd3982c05820
#owner: https://api.github.com/users/sstock2005

sudo dpkg -r mullvad-vpn # if you have it installed already
echo "deb [signed-by=/usr/share/keyrings/mullvad-keyring.asc arch=$( dpkg --print-architecture )] https://repository.mullvad.net/deb/stable testing main" | sudo tee /etc/apt/sources.list.d/mullvad.list
sudo apt update
sudo apt install mullvad-vpn