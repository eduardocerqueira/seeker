#date: 2023-01-24T16:45:21Z
#url: https://api.github.com/gists/471355ccabf0a3685c29bcdf0fcbdcb6
#owner: https://api.github.com/users/holzi1005

#!/bin/bash

# Use the script to setup the inferfaces on vm with cloud init
# The script will be loaded to the server and setup the inferfaces.
# After the execution, the server need to be restarted.

echo  "" >> /etc/network/interfaces
echo  "" >> /etc/network/interfaces
echo  "" >> /etc/network/interfaces
echo "auto $1" >> /etc/network/interfaces
echo "iface $1 inet static" >> /etc/network/interfaces
echo "  address $2" >> /etc/network/interfaces
if [ -z $3 ]; then
echo "  gateway $3" >> /etc/network/interfaces
fi