#date: 2021-10-29T16:45:33Z
#url: https://api.github.com/gists/1385e696ff6bbd31415ad36c7a87a10a
#owner: https://api.github.com/users/rajathithan

#!/bin/bash
#######################################################################
#
# Author        : Rajathithan Rajasekar
# Date          : 10/28/2021
# Script Usage  : Sql-Port-FWDR MIGS Startup script
#
#######################################################################

# General updates
echo "****************************************************************"
echo "Updating the server:"
echo "****************************************************************"
sudo apt-get -y update
sudo apt-get -y install net-tools
sudo apt-get -y upgrade google-guest-agent
sudo timedatectl set-timezone America/Chicago


#variables
cloudsqlprivateip='someprivateip:3306'
dport=4479
ipaddress=$(/sbin/ifconfig ens4 | grep 'inet addr' | cut -d: -f2| awk '{print $1}')
line='net.ipv4.ip_forward=1'
BUCKET=somebucketname


# Update the /etc/sysctl.conf for port forwarding
echo "****************************************************************"
echo "Update the /etc/sysctl.conf for port forwarding:"
echo "****************************************************************"
sudo sed -i "/^#$line/ c$line" /etc/sysctl.conf
sudo sed -i -e '$anet.ipv4.conf.all.route_localnet=1' /etc/sysctl.conf
sudo sysctl -p


# Update the iptables & create logging
echo "****************************************************************"
echo "Update the iptables & create logging:"
echo "****************************************************************"

sudo iptables -F
sudo iptables -F -t nat
sudo iptables -t nat -A PREROUTING -p tcp -d $ipaddress --dport $dport -j DNAT --to-destination $cloudsqlprivateip
sudo iptables -t nat -A OUTPUT -o lo -d 127.0.0.1 -p tcp --dport $dport -j DNAT --to-destination $cloudsqlprivateip
sudo iptables -t nat -A OUTPUT -o lo -d $ipaddress -p tcp --dport $dport -j DNAT --to-destination $cloudsqlprivateip
sudo iptables -t nat -A POSTROUTING -j MASQUERADE
sudo iptables -L -t nat
# Enable Logging
sudo iptables -A INPUT -p tcp --dport $dport --syn -j LOG --log-prefix "iptables: " --log-level 7
sudo iptables -A FORWARD -p tcp --dport $dport --syn -j LOG --log-prefix "iptables: " --log-level 7
sudo iptables -A OUTPUT -p tcp --dport $dport --syn -j LOG --log-prefix "iptables: " --log-level 7
# Save Iptables rules
sudo sh -c "iptables-save > /etc/iptables.rules"



# Configure rsyslog for iptables logging
echo "****************************************************************"
echo "Creating the startup script:"
echo "****************************************************************"
sudo cat <<EOF >> 10-iptables.conf
:msg, startswith, "iptables: " -/var/log/iptables/iptables.log
& ~
:msg, regex, "^\[ *[0-9]*\.[0-9]*\] iptables: " -/var/log/iptables/iptables.log
& ~
EOF

# Move the .conf file to /etc/rsyslog.d/
echo "****************************************************************"
echo "Moving the .conf file to /etc/rsyslog.d/:"
echo "****************************************************************"
sudo mv 10-iptables.conf /etc/rsyslog.d/10-iptables.conf
sudo chown root:root /etc/rsyslog.d/10-iptables.conf



# Create the log rotation
echo "****************************************************************"
echo "Creating the log rotation for port-fwdr:"
echo "****************************************************************"
sudo cat <<EOF >> iptables
/var/log/iptables/iptables.log
{
    rotate 7
    daily
    missingok
    notifempty
    delaycompress
    compress
    postrotate
    invoke-rc.d rsyslog rotate > /dev/null
    endscript
}
EOF

# Move the log rotation file to /etc/logrotate.d
echo "****************************************************************"
echo "Moving the cloud_sql_proxy_log_rotate to /etc/logrotate.d/:"
echo "****************************************************************"
sudo mv iptables /etc/logrotate.d/iptables
sudo chown root:root /etc/logrotate.d/iptables


# restart the required services
echo "****************************************************************"
echo "Restarting the system services:"
echo "****************************************************************"
sudo service rsyslog restart
sudo service systemd-networkd restart

# Install fluentd
echo "****************************************************************"
echo "Installing the fluentd services:"
echo "****************************************************************"
wget https://dl.google.com/cloudagents/add-logging-agent-repo.sh -O add-logging-agent-repo.sh
FILE=add-logging-agent-repo.sh
if [ -f "$FILE" ]; then
    echo "$FILE exists."
    sudo bash add-logging-agent-repo.sh --also-install
else
    echo "$FILE does not exist."
    sudo gsutil cp gs://${BUCKET}/add-logging-agent-repo.sh add-logging-agent-repo.sh
    sudo bash add-logging-agent-repo.sh --also-install
fi


# configure fluentd iptables.conf for stackdriver export
echo "****************************************************************"
echo "Configuring the fluentd iptables.conf:"
echo "****************************************************************"
sudo cat <<EOF >> iptables.conf
<source>
 @type tail

 # Parse the timestamp, but still collect the entire line as 'message'
 format /^(?<message>(?<time>[^ ]*\s*[^ ]* [^ ]*) .*)$/

 path /var/log/iptables/iptables.log
 pos_file /var/lib/google-fluentd/pos/iptables.pos
 read_from_head true
 tag iptables
</source>
EOF

# Move the config file under /etc/google-fluentd/config.d/
echo "****************************************************************"
echo "Moving iptables.conf to /etc/google-fluentd/config.d/:"
echo "****************************************************************"
sudo mv iptables.conf /etc/google-fluentd/config.d/iptables.conf
sudo chown root:root /etc/google-fluentd/config.d/iptables.conf
