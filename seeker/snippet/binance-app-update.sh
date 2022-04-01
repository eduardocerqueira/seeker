#date: 2022-04-01T16:48:58Z
#url: https://api.github.com/gists/f23aa4cbe85d2be711cdacf04da20d53
#owner: https://api.github.com/users/tatarysh

TEMP_DEB="$(mktemp)" &&
wget -O "$TEMP_DEB" 'https://ftp.binance.com/electron-desktop/linux/production/binance-amd64-linux.deb' &&
sudo dpkg -i "$TEMP_DEB"
rm -f "$TEMP_DEB"
