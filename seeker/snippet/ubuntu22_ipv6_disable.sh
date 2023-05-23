#date: 2023-05-23T16:56:42Z
#url: https://api.github.com/gists/835fb585055e47ca332eba52a4190d02
#owner: https://api.github.com/users/joshespi

#!/bin/bash

# Append IPv6 disable configuration to /etc/sysctl.conf
echo "net.ipv6.conf.all.disable_ipv6=1
net.ipv6.conf.default.disable_ipv6=1
net.ipv6.conf.lo.disable_ipv6=1" | sudo tee -a /etc/sysctl.conf > /dev/null

# Apply the changes
sudo sysctl -p

# Create /etc/rc.local if it doesn't exist
if [ ! -f /etc/rc.local ]; then
    sudo touch /etc/rc.local
    sudo chmod +x /etc/rc.local
    echo "#!/bin/bash" | sudo tee /etc/rc.local > /dev/null
    echo "# /etc/rc.local" | sudo tee -a /etc/rc.local > /dev/null
fi

# Add commands to /etc/rc.local
sudo sed -i '/exit 0/d' /etc/rc.local
echo "/etc/sysctl.d" | sudo tee -a /etc/rc.local > /dev/null
echo "/etc/init.d/procps restart" | sudo tee -a /etc/rc.local > /dev/null
echo "exit 0" | sudo tee -a /etc/rc.local > /dev/null

# Change permission of /etc/rc.local
sudo chmod 755 /etc/rc.local
