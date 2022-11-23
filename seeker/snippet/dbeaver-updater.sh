#date: 2022-11-23T17:05:08Z
#url: https://api.github.com/gists/aa6dc8124d196df6d676e77431ecbfe4
#owner: https://api.github.com/users/renanstn

wget -O dbeaver.deb https://dbeaver.io/files/dbeaver-ce_latest_amd64.deb
chmod +x dbeaver.deb
sudo dpkg -i dbeaver.deb
rm dbeaver.deb
