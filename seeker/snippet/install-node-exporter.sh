#date: 2025-04-30T16:51:25Z
#url: https://api.github.com/gists/afbe72ad0d2db5e89b387329a252372c
#owner: https://api.github.com/users/alexeynavarkin

#!/bin/bash
set -euxo pipefail

version='1.9.1'
arch='linux-amd64'
install_path='/usr/local/bin'
download_url="https://github.com/prometheus/node_exporter/releases/download/v${version}/node_exporter-${version}.${arch}.tar.gz"

id -u node_exporter &>/dev/null ||  sudo useradd --no-create-home --shell /bin/false node_exporter

mkdir -p tmp

curl -s -L ${download_url} > tmp/node_exporter.tar.gz
tar -xvzf tmp/node_exporter.tar.gz -C tmp --strip-components=1

cp tmp/node_exporter ${install_path}/node_exporter

cat > /etc/systemd/system/node_exporter.service <<EOL
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=${install_path}/node_exporter

[Install]
WantedBy=multi-user.target
EOL

systemctl daemon-reload
systemctl enable node_exporter
systemctl start node_exporter

rm -rf tmp