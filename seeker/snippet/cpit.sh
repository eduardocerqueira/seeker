#date: 2022-05-20T17:15:35Z
#url: https://api.github.com/gists/213239dd0fa3afec8982032774dfd64b
#owner: https://api.github.com/users/phantomic12

sudo apt update
sudo apt install cockpit -y
sudo systemctl start cockpit
sudo systemctl status cockpit
sudo ufw allow 9090/tcp