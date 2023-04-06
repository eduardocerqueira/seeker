#date: 2023-04-06T16:44:55Z
#url: https://api.github.com/gists/a11f6bc78571fb7a2e7989dd3d957905
#owner: https://api.github.com/users/brunobritodev

# Don't you hate when you need to root to fix dumb issues?
sudo service network-manager stop
sudo rm /var/lib/NetworkManager/NetworkManager.state
sudo service network-manager start
# reboot -h now