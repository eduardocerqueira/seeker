#date: 2025-01-14T16:41:42Z
#url: https://api.github.com/gists/b0a89266f818d363ab05e25a72b76032
#owner: https://api.github.com/users/coltenkrauter

midclt call vm.query | jq -r '
  .[] | 
  select(.status.state=="RUNNING") | 
  .devices[] | 
  select(.dtype=="NIC") | 
  .attributes.mac' | while read -r mac; do 
  echo -n "$mac: " 
  arp -a | grep -i "$mac" | awk '{print $1, $2}' 
done
