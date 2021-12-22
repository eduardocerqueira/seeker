#date: 2021-12-22T17:05:24Z
#url: https://api.github.com/gists/af7a24c6c668ca26a9658c61af766eca
#owner: https://api.github.com/users/gscho

#!/bin/bash

GENERATED_PASSWORD=$(openssl rand -hex 16)
PASSWORD=${COLLECTD_PASSWORD:-$GENERATED_PASSWORD}
USERNAME=${COLLECTD_USERNAME:-ghes0}
NTP_SERVER=${COLLECTD_NTP_SERVER:-0.github.pool.ntp.org}

echo
echo "Installing collectd packages"

sudo apt-get update
sudo apt install -y collectd collectd-utils ntpdate apache2-utils

echo
echo "Using ntp server: $NTP_SERVER"

### This is to time sync the GHES appliance and this server.
sudo ntpdate $NTP_SERVER

sudo tee /etc/cron.daily/ntp-sync > /dev/null << EOF
  #!/bin/sh
  ntpdate $NTP_SERVER
EOF

### htpassword is used to create a passwd file with a username and hashed password delimited by a colon.
htpasswd -b -c /etc/collectd/passwd $USERNAME $PASSWORD

### Sets up a logfile. Modify the "File" parameter below to change the name or location of the log.
sudo tee /etc/collectd/collectd.conf.d/logfile.conf > /dev/null << EOF
  LoadPlugin logfile

  <Plugin logfile>
          LogLevel "info"
          File "/var/log/collectd.log"
          Timestamp true
          PrintSeverity false
  </Plugin>
EOF

### Configures collectd to listen on UDP:25826 and requre encrypted traffic. The port parameter can be changed as needed.
sudo tee /etc/collectd/collectd.conf.d/network.conf > /dev/null << EOF
  LoadPlugin network

  <Plugin network>
    <Listen "0.0.0.0" "25826">
            SecurityLevel "Encrypt"
            AuthFile "/etc/collectd/passwd"
    </Listen>
  </Plugin>
EOF

### Configured collectd to start a webserver on TCP:9103 and write metrics in a prometheus compatible format.
sudo tee /etc/collectd/collectd.conf.d/prometheus.conf > /dev/null << EOF
  LoadPlugin write_prometheus
  <Plugin write_prometheus>
          Port "9103"
  </Plugin>
EOF

echo
echo "Restarting collectd service..."
sudo systemctl restart collectd

### The username and password printed below are needed to configure the collectd forwarding client on the GHES server.
echo
echo "COLLECTD_USERNAME:COLLECTD_PASSWORD"
cat /etc/collectd/passwd