#date: 2025-01-27T17:07:14Z
#url: https://api.github.com/gists/6ad9ecf17622529d237cc1a0bc22c819
#owner: https://api.github.com/users/shiwildy

# >> Block Torrent Port
iptables -A FORWARD -p tcp --dport 6881:6889 -j DROP
iptables -A FORWARD -p udp --dport 6881:6889 -j DROP
iptables -A FORWARD -p tcp --dport 6969 -j DROP
iptables -A FORWARD -p udp --dport 6969 -j DROP
iptables -A FORWARD -p tcp --dport 51413 -j DROP
iptables -A FORWARD -p udp --dport 51413 -j DROP
iptables -A FORWARD -p tcp --dport 27014:27050 -j DROP
iptables -A FORWARD -p udp --dport 27014:27050 -j DROP
iptables -A FORWARD -p udp --dport 4444 -j DROP
iptables -A FORWARD -p udp --dport 51413 -j DROP
iptables -A FORWARD -p udp --dport 8999 -j DROP
iptables -A FORWARD -p udp --dport 8000:9000 -j DROP

# >> Block all magnet and high udp traffic
iptables -A FORWARD -p tcp --dport 80 -m string --algo bm --string "magnet:?" -j DROP
iptables -A FORWARD -p tcp --dport 443 -m string --algo bm --string "magnet:?" -j DROP
iptables -A FORWARD -p udp -m length --length 80:65535 -j DROP
iptables -A FORWARD -m string --algo bm --string "BitTorrent" -j LOG --log-prefix "BLOCKED TORRENT: "
iptables -A FORWARD -m string --algo bm --string "magnet:?" -j LOG --log-prefix "BLOCKED MAGNET: "

# >> Create LOGDROP
iptables -N LOGDROP 2>/dev/null
iptables -F LOGDROP
iptables -A LOGDROP -j DROP

# >> Block Torrent By strings
iptables -A FORWARD -m string --algo bm --string "BitTorrent" -j LOGDROP
iptables -A FORWARD -m string --algo bm --string "BitTorrent protocol" -j LOGDROP
iptables -A FORWARD -m string --algo bm --string "peer_id=" -j LOGDROP
iptables -A FORWARD -m string --algo bm --string ".torrent" -j LOGDROP
iptables -A FORWARD -m string --algo bm --string "announce.php?passkey=" -j LOGDROP
iptables -A FORWARD -m string --algo bm --string "torrent" -j LOGDROP
iptables -A FORWARD -m string --algo bm --string "announce" -j LOGDROP
iptables -A FORWARD -m string --algo bm --string "info_hash" -j LOGDROP
iptables -A FORWARD -m string --algo bm --string "tracker" -j LOGDROP

# DHT keywords
iptables -A FORWARD -m string --algo bm --string "get_peers" -j LOGDROP
iptables -A FORWARD -m string --algo bm --string "announce_peer" -j LOGDROP
iptables -A FORWARD -m string --algo bm --string "find_node" -j LOGDROP
iptables -A FORWARD -m string --algo bm --string "magnet:?" -j LOGDROP

# >> Block ALL Input Chain
iptables -A INPUT -m string --algo bm --string "BitTorrent" -j DROP
iptables -A INPUT -m string --algo bm --string "BitTorrent protocol" -j DROP
iptables -A INPUT -m string --algo bm --string "peer_id=" -j DROP
iptables -A INPUT -m string --algo bm --string ".torrent" -j DROP
iptables -A INPUT -m string --algo bm --string "announce.php?passkey=" -j DROP
iptables -A INPUT -m string --algo bm --string "torrent" -j DROP
iptables -A INPUT -m string --algo bm --string "announce" -j DROP
iptables -A INPUT -m string --algo bm --string "info_hash" -j DROP
iptables -A INPUT -m string --algo bm --string "tracker" -j DROP

# >> Block ALL Output Chain
iptables -A OUTPUT -m string --algo bm --string "BitTorrent" -j DROP
iptables -A OUTPUT -m string --algo bm --string "BitTorrent protocol" -j DROP
iptables -A OUTPUT -m string --algo bm --string "peer_id=" -j DROP
iptables -A OUTPUT -m string --algo bm --string ".torrent" -j DROP
iptables -A OUTPUT -m string --algo bm --string "announce.php?passkey=" -j DROP
iptables -A OUTPUT -m string --algo bm --string "torrent" -j DROP
iptables -A OUTPUT -m string --algo bm --string "announce" -j DROP
iptables -A OUTPUT -m string --algo bm --string "info_hash" -j DROP
iptables -A OUTPUT -m string --algo bm --string "tracker" -j DROP