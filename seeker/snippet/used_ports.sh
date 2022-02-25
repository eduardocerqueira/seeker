#date: 2022-02-25T17:07:47Z
#url: https://api.github.com/gists/9c47bd440db52fbdd30ac46bd8e1f403
#owner: https://api.github.com/users/FredyRosero

sudo netstat -tulpn | grep LISTEN
# -t --tcp 
# -u --udp 
# -l Show only listening sockets. (These are omitted by default.) 
# -p Show the PID and name of the program to which each socket belongs.
# -n Show numerical addresses instead of trying to determine symbolic host, port or user names. 

sudo nmap -sTU -O $IP
# -sT (TCP connect scan) 
#     TCP connect scan is the default TCP scan type when SYN scan is not an option. This is the case when a user 
#     does not have raw packet privileges or is scanning IPv6 networks. Instead of writing raw packets as 
#     most other scan types do, Nmap asks the underlying operating system to establish a connection with 
#     the target machine and port by issuing the connect system call. This is the same high-level system call 
#     that web browsers, P2P clients, and most other network-enabled applications use to establish a connection.  
# -sS (TCP SYN scan): 
#     SYN scan is the default and most popular scan option for good reasons. It can be performed quickly, 
#     scanning thousands of ports per second on a fast network not hampered by restrictive firewalls. 
#     It is also relatively unobtrusive and stealthy since it never completes TCP connections. 
#     SYN scan works against any compliant TCP stack rather than depending on idiosyncrasies of specific platforms 
#     as Nmap's FIN/NULL/Xmas, Maimon and idle scans do. It also allows clear, reliable differentiation between the open, 
#     closed, and filtered states.
# -sU (UDP scans)
#     While most popular services on the Internet run over the TCP protocol, UDP [6] services are widely deployed. DNS, SNMP, 
#     and DHCP (registered ports 53, 161/162, and 67/68) are three of the most common. Because UDP scanning is generally 
#     slower and more difficult than TCP, some security auditors ignore these ports. This is a mistake, as exploitable UDP 
#     services are quite common and attackers certainly don't ignore the whole protocol. 
#     Fortunately, Nmap can help inventory UDP ports.
# -O  (Enable OS detection)
#     Enables OS detection, as discussed above. Alternatively, you can use -A to enable OS detection along with other things. 