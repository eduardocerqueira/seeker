#date: 2025-08-19T17:04:17Z
#url: https://api.github.com/gists/6e4bed77f68164e9bc5b2474e46055e1
#owner: https://api.github.com/users/Mqxx

#!/bin/bash
set -e

echo "Disabling cloud init services..."
echo "cloud-init cloud-init/datasources multiselect None -NoCloud -ConfigDrive -OpenNebula -DigitalOcean -Azure -AltCloud -OVF -MAAS -GCE -OpenStack -CloudSigma -SmartOS -Bigstep -Scaleway -AliYun -Ec2 -CloudStack -Hetzner -IBMCloud -Oracle -Exoscale -RbxCloud -UpCloud -VMware -Vultr -LXD -NWCS -Akamai" | sudo debconf-set-selections


echo "Purging cloud init package..."
sudo apt-get purge -y cloud-init

echo "Deleting cloud init files..."
sudo rm -rf /etc/cloud/ /var/lib/cloud/

echo "Rebooting..."
sudo reboot
