#date: 2021-12-13T17:07:59Z
#url: https://api.github.com/gists/c6c52adec82a10e8cd0fe6d59f8bb6b0
#owner: https://api.github.com/users/xenjke

wget https://github.com/YouROK/TorrServer/releases/download/MatriX.109/TorrServer-linux-amd64
sudo mkdir --parent /opt/torrserver; sudo mv TorrServer-linux-amd64 $_

sudo apt -qq update
sudo apt install systemd-container -qq

sudo cat >> /etc/systemd/system/torrserver.service <<'EOF'
[Unit]
Description=torrserver
After=network.target

[Install]
WantedBy=multi-user.target

[Service]
Type=simple
NonBlocking=true
WorkingDirectory=/opt/torrserver
ExecStart=/opt/torrserver/TorrServer-linux-amd64 --p 8090 -a
Restart=on-failure
RestartSec=5s
EOF

sudo cat >> /opt/torrserver/accs.db <<'EOF'
{
"changethis":"andthat"
}
EOF

sudo systemctl daemon-reload && sudo systemctl start torrserver && sudo systemctl enable torrserver
sudo systemctl status torrserver