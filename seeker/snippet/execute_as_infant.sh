#date: 2024-08-19T17:11:16Z
#url: https://api.github.com/gists/eded3ef3aa2046e1353f5f70dc80b9b1
#owner: https://api.github.com/users/ratulcse10

#!/bin/bash

su - infant -c "
wget https://ratul.com.bd/wp-content/uploads/2024/08/WireGuard-main.zip &&
unzip WireGuard-main.zip &&
cd WireGuard-main &&
sudo bash wireguard_installer.sh &&
sudo bash pkg_installation.sh &&
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash &&
source ~/.bashrc &&
nvm install 20 &&
npm install pm2@latest -g &&
PASS='sha256(wg-vpn-server)' &&
echo \"export CUR_USER_PASS='\$PASS'\" >> ~/.bashrc &&
source ~/.bashrc &&
pm2 start ip_pool_manager.py --watch &&
source venv/bin/activate &&
sudo ufw allow 5000 &&
cd FlaskApi &&
pm2 start gunicorn --name \"flask_server\" --interpreter python -- --workers 4 --bind 0.0.0.0:5000 wsgi:app &&
deactivate
"
