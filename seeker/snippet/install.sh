#date: 2024-03-11T16:54:48Z
#url: https://api.github.com/gists/27e7b309f6c1590a4eab783871fb8065
#owner: https://api.github.com/users/isu-kim

echo "**** install runtime dependencies ****"
sudo sed -i 's/archive.ubuntu.com/ftp.kaist.ac.kr/g' /etc/apt/sources.list
sudo apt-get update
sudo apt-get install -y \
curl \
jq \
libatomic1 \
net-tools \
netcat \
python3-pip \
gdb \
wget

echo "**** clean up ****"
sudo apt-get clean
sudo rm -rf \
/config/* \
/tmp/* \
/var/lib/apt/lists/* \
/var/tmp/*

echo "**** setting code-server ****"
sudo -u koicloud mkdir -p /home/koicloud/vscode/
sudo -u koicloud mkdir -p /home/koicloud/.config/code-server/
sudo -u koicloud /app/code-server/bin/code-server --install-extension /app/code-server/extension/cpptools-linux.vsix
sudo -u koicloud sed -i "s/127.0.0.1:8080/0.0.0.0:8443/g" /home/koicloud/.config/code-server/config.yaml      
sudo -u koicloud sed -i "3s/.*/password: "**********"
sudo cat /home/koicloud/.config/code-server/config.yaml
sudo sh -c 'printf "[Unit]\nDescription=code-server\nAfter=nginx.service\n[Service]\nType=simple\nEnvironment=PASSWORD=\$PASSWORD\nExecStart=/usr/bin/sudo -u koicloud /app/code-server/bin/code-server /home/koicloud/vscode --bind-addr 0.0.0.0: "**********"
sudo -u koicloud cat  /usr/lib/systemd/system/code-server.service
sudo -u koicloud sed -i 's/passwordMsg = \"Password was set from \$PASSWORD.\";/passwordMsg = \"Visit <a href=\\"https: "**********"
sudo -u koicloud sed -i 's/let passwordMsg = `Check the config file at ${(0, util_1.humanPath)(os.homedir(), req.args.config)} for the password.`;/let passwordMsg = \"Visit <a href=\\"https: "**********"
sudo sed -i "s/\/\"/:\/app\/code-server\/bin\//g" /etc/environment

sudo systemctl enable code-server
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/* /var/tmp/*
sudo systemctl restart code-serverhomedir(), req.args.config)} for the password.`;/let passwordMsg = \"Visit <a href=\\"https: "**********"
sudo sed -i "s/\/\"/:\/app\/code-server\/bin\//g" /etc/environment

sudo systemctl enable code-server
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/* /var/tmp/*
sudo systemctl restart code-server clean
sudo rm -rf /var/lib/apt/lists/* /var/tmp/*
sudo systemctl restart code-server