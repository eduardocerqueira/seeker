#date: 2021-11-05T16:54:56Z
#url: https://api.github.com/gists/a55573fcd199b4f3912a419c81827702
#owner: https://api.github.com/users/Hytreenee

############

# Useful links

############


# main info source
# https://shadowsocks.org/en/config/quick-guide.html

# main client used here "shadowsocks-libev"
# https://github.com/shadowsocks/shadowsocks-libev

# another client, not from guide
# https://github.com/shadowsocks/shadowsocks-qt5/wiki/Installation

# alt guide
# https://www.oilandfish.com/posts/shadowsocks-libev.html



############

# Server

############


sudo apt install -y snapd
sudo snap install core
sudo snap install shadowsocks-libev
sudo snap alias shadowsocks-libev.ss-local ss-local
sudo snap alias shadowsocks-libev.ss-redir ss-redir
sudo snap alias shadowsocks-libev.ss-server ss-server
sudo snap alias shadowsocks-libev.ss-tunnel ss-tunnel
sudo snap alias shadowsocks-libev.ss-manager ss-manager
sudo mkdir -p /var/snap/shadowsocks-libev/common/etc/shadowsocks-libev
sudo touch /var/snap/shadowsocks-libev/common/etc/shadowsocks-libev/config.json

sudo bash -c 'cat <<EOT >/var/snap/shadowsocks-libev/common/etc/shadowsocks-libev/config.json
{
    "server":"server_ip",
    "server_port":PORT,
    "local_port":1080,
    "password":"pass",
    "timeout":20,
    "method":"chacha20-ietf-poly1305",
    "nameserver":"1.1.1.1",
    "mode":"tcp_and_udp"
}
EOT
'

sudo touch /etc/systemd/system/shadowsocks-libev-server.service

sudo bash -c 'cat <<EOT >/etc/systemd/system/shadowsocks-libev-server.service
[Unit]
Description=Shadowsocks-Libev Server
After=network.target
After=network-online.target

[Service]
Type=simple
ExecStart=/snap/bin/ss-server -c /var/snap/shadowsocks-libev/common/etc/shadowsocks-libev/config.json

[Install]
WantedBy=multi-user.target
EOT
'

sudo ufw allow PORT
sudo systemctl enable --now shadowsocks-libev-server
sudo systemctl status shadowsocks-libev-server



############

# Server | Additional tweaks

############


sudo bash -c 'cat <<EOT >>/etc/sysctl.conf
#
fs.file-max = 51200
net.core.netdev_max_backlog = 250000
net.core.somaxconn = 4096
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_tw_recycle = 0
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 1200
net.ipv4.ip_local_port_range = 10000 65000
net.core.netdev_max_backlog = 4096
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.tcp_max_tw_buckets = 5000
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_mtu_probing = 1
net.core.rmem_max = 67108864
net.core.wmem_max = 67108864
net.ipv4.tcp_mem = 25600 51200 102400
net.ipv4.tcp_rmem = 4096 87380 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864
EOT
'

sudo sysctl -p



############

# Client

############


sudo snap install shadowsocks-libev
sudo snap alias shadowsocks-libev.ss-local ss-local
sudo snap alias shadowsocks-libev.ss-redir ss-redir
sudo snap alias shadowsocks-libev.ss-server ss-server
sudo snap alias shadowsocks-libev.ss-tunnel ss-tunnel
sudo snap alias shadowsocks-libev.ss-manager ss-manager
sudo ss-local -c path/to/same_config.json

# Now you can connect in browser
# Use socks5 + "Proxy DNS when using SOCKS v5" checkbox



############

# Client | Auto start on boot

############


# Same config file as for server
sudo touch /var/snap/shadowsocks-libev/common/config.json

sudo bash -c 'cat <<EOT >/var/snap/shadowsocks-libev/common/config.json
{
    "server":"server_ip",
    "server_port":PORT,
    "local_port":1080,
    "password":"pass",
    "timeout":20,
    "method":"chacha20-ietf-poly1305",
    "nameserver":"1.1.1.1",
    "mode":"tcp_and_udp"
}
EOT
'

sudo touch /etc/systemd/system/shadowsocks-libev-client.service

sudo bash -c 'cat <<EOT >/etc/systemd/system/shadowsocks-libev-client.service
[Unit]
Description=ShadowSocks Client
After=network.target
After=network-online.target

[Service]
ExecStart=/snap/bin/ss-local -c /var/snap/shadowsocks-libev/common/config.json
Restart=on-failure

[Install]
WantedBy=multi-user.target 
EOT
'

sudo systemctl enable --now shadowsocks-libev-client
sudo systemctl status shadowsocks-libev-client



############

# Other

############

# After all the steps you now able to set 127.0.0.1:1080 as a socks5 proxy in any software

# To create connection LINK (for some not mentioned clients required) use "btoa" in browser console
# console.log( "ss://" + btoa("chacha20-ietf-poly1305:PASS@IP:PORT") )
