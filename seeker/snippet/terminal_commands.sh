#date: 2025-11-24T17:06:28Z
#url: https://api.github.com/gists/a92d45ada3a7ac1335606215493c7e94
#owner: https://api.github.com/users/jdavidrcamacho

sudo ufw status
sudo ufw allow ssh
sudo ufw enable
sudo ufw deny from 10.50.20.8
sudo ufw deny out to 10.50.20.8
sudo ufw deny from 10.50.20.6
sudo ufw deny out to 10.50.20.6
sudo ufw reload