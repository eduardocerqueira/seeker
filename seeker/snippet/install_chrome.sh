#date: 2024-06-20T16:44:51Z
#url: https://api.github.com/gists/e4726077ee927ba888c6b7b46470b451
#owner: https://api.github.com/users/bharathibh

wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt-get install ./google-chrome-stable_current_amd64.deb -y && apt-get clean
apt-get install -f -y && apt-get clean

google-chrome --product-version