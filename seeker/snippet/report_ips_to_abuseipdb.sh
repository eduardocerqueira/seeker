#date: 2023-11-01T17:08:58Z
#url: https://api.github.com/gists/5b9c93f8354f2a41ac94f09e3a38c517
#owner: https://api.github.com/users/ramit-mitra

#!/bin/bash

# Example: curl -sk https://gist.githubusercontent.com/ramit-mitra/f72b16b34099f85b3423bd32f63930c3/raw/ufw_block_banned_ips.sh | bash

url="https://raw.githubusercontent.com/ramit-mitra/blocklist-ipsets/main/rottenIPs.json"

echo "-------------------------"
echo "Reading Banlist IPs..."

blocked_ips=$(curl -s $url | jq -r '.[]')

echo "Banning IPs (using UFW)..."
echo "-------------------------"
	
for ip in $blocked_ips; do
  echo $ip
  sudo ufw deny from $ip to any > /dev/null
done

echo "-------------------------"
sudo ufw reload
sudo ufw status
echo "Done"