#date: 2023-06-12T16:47:05Z
#url: https://api.github.com/gists/770c3f3c8e9e4440a99451b34221fbbf
#owner: https://api.github.com/users/ibshafique

#!/bin/bash

#==============================================================================#
#title           :prometheus.sh                                                #
#description     :This Bash script will automatically install Prometheus       #
#date            :12-06-2023                                                   #
#email           :ibshafique@gmal.com                                          #
#==============================================================================#

# Check if the script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root."
  exit 1
fi

# Install required packages
yum update -y
yum install -y wget

# Prerequisites 
adduser -M -r -s /sbin/nologin prometheus
mkdir /var/lib/prometheus
mkdir /etc/prometheus/

# Download Prometheus
cd /tmp/
wget https://github.com/prometheus/prometheus/releases/download/v2.36.2/prometheus-2.36.2.linux-amd64.tar.gz

# Extract the downloaded tarball
tar xzf prometheus-2.36.2.linux-amd64.tar.gz

# Copy some required files
cp prometheus-2.36.2.linux-amd64/prometheus /usr/local/bin/
cp prometheus-2.36.2.linux-amd64/promtool /usr/local/bin/
cp -r prometheus-2.36.2.linux-amd64/consoles /etc/prometheus
cp -r prometheus-2.36.2.linux-amd64/console_libraries /etc/prometheus/
cp prometheus-2.36.2.linux-amd64/prometheus.yml /etc/prometheus

echo "Setting up the prometheus.yml config file"
cat << EOF > /etc/prometheus/prometheus.yml
# my global config
global:
  scrape_interval: 15s # Set the scrape interval to every 15 seconds. Default is every 1 minute.
  evaluation_interval: 15s # Evaluate rules every 15 seconds. The default is every 1 minute.
  # scrape_timeout is set to the global default (10s).

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

# Load rules once and periodically evaluate them according to the global 'evaluation_interval'.
rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

# A scrape configuration containing exactly one endpoint to scrape:
# Here it's Prometheus itself.
scrape_configs:
  # The job name is added as a label `job=<job_name>` to any timeseries scraped from this config.
  - job_name: "prometheus"

    # metrics_path defaults to '/metrics'
    # scheme defaults to 'http'.

    static_configs:
      - targets: ["192.168.6.107:9090"]
EOF

# Changing ownership of prometheus directory
chown prometheus:prometheus /etc/prometheus
chown prometheus:prometheus /var/lib/prometheus

#create a systemd service file to manage the Prometheus service via systemd
echo "creating systemd service"
cat << EOF > /etc/systemd/system/prometheus.service
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \
    --config.file /etc/prometheus/prometheus.yml \
    --storage.tsdb.path /var/lib/prometheus/ \
    --web.console.templates=/etc/prometheus/consoles \
    --web.console.libraries=/etc/prometheus/console_libraries

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now prometheus