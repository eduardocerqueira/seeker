#date: 2023-12-08T16:40:55Z
#url: https://api.github.com/gists/e541616d03126a21678cb395f0b52d73
#owner: https://api.github.com/users/VMatyagin

#  This script is adapted/tweaked from the openWRT wiki page on creating VPN server.
#  VPN client can access outside world as if the traffic originates from the openWRT router.
#
#  Prerequisites
#  1. opkg update && opkg install openvpn-openssl openvpn-easy-rsa
#  2. Get a public DDNS domain name or a static IP for the vpn server, put it into ddns_name="" near the bottom of the script.
#  3. Customize parameters, server/client service name, subnet, server port, output dir etc in the same bottom section.
#
#  USAGE:
#  1. sh ./ovpn_owrt.sh <pki directory> [optional dh.pem file]
#  2. run /etc/openvpn/stop && sleep 10 && /etc/openvpn/start
#  3. install or import /tmp/test_ovpnout/<client_name>.ovpn into the device's OVPN client software.
# 
#  The pki directory is where certificates are created and managed, often people would use /etc/easyrsa/pki.
#  Once generated, this script will *not* overwrite existing files if re-run with the same parameters.
#  Therefore, to generate more client keys, change the client_name, and invoke with the same pki-dir value should do.
#
#  User can supply a dh.pem generated on another faster machine, it may taka minutes to generate on the router.
#  To generate a dh.pem: "openssl dhparam -dsaparam -out dh.pem 4096"
# 
#  OUTPUT: /etc/openvpn/server_<port>.conf, with keys/certs embedded.
#          /tmp/test_ovpn/client_<port>.ovpn, can be imported/installed on the vpn client device.
#

fw_subnet_snat() {     # SNAT outbound VPN client traffic to look like from this router (src_ip).
    local subnet="$1"  # subnet whose traffic are to be SNAT'ed
    local src_ip="$2"  # external IP of the router
    local output="$3"  # output firewall rules file
    # Create an iptable routing rule in a file, then tell firewall to include it.
    cat << EOF > $output
iptables -I INPUT -i tun+ -j ACCEPT
iptables -I FORWARD -i tun+ -j ACCEPT
iptables -I OUTPUT -o tun+ -j ACCEPT
iptables -I FORWARD -o tun+ -j ACCEPT
iptables -t nat -A POSTROUTING -s ${subnet}/24 -j SNAT --to-source ${src_ip}
EOF
}

run_easyrsa() {
    if test -z $EASYRSA_PKI; then
        echo "MUST PROVIDE output PKI directory."
        exit 1
    fi

    local dh_file=$1
    if ! test -e $EASYRSA_PKI; then
        echo "==> Initializing new easy-rsa PKI directory $EASYRSA_PKI for certificates."
        easyrsa init-pki
    else
        echo "==> Using existing easy-rsa PKI directory $EASYRSA_PKI."
    fi

    [ ! -e ${EASYRSA_PKI}/dh.pem ] && {
        if ! test -z $dh_file && test -e $dh_file; then
            echo "==> Using user-supplied DH.pem file: $dh_file."
            cp $dh_file ${EASYRSA_PKI}/dh.pem
        else # "openssl dhparam -dsaparam" is much faster than "easyrsa gen-dh"
            echo "==> Building DH.pem ..."
            openssl dhparam -dsaparam -out ${EASYRSA_PKI}/dh.pem $EASYRSA_KEY_SIZE
        fi
    }
    [ ! -e ${EASYRSA_PKI}/ca.crt ] && echo "==> Building ca.crt" && easyrsa build-ca nopass # default no passphrase
 "**********"  "**********"  "**********"  "**********"  "**********"[ "**********"  "**********"! "**********"  "**********"- "**********"e "**********"  "**********"$ "**********"{ "**********"E "**********"A "**********"S "**********"Y "**********"R "**********"S "**********"A "**********"_ "**********"P "**********"K "**********"I "**********"} "**********"/ "**********"t "**********"c "**********". "**********"p "**********"e "**********"m "**********"  "**********"] "**********"  "**********"& "**********"& "**********"  "**********"e "**********"c "**********"h "**********"o "**********"  "**********"" "**********"= "**********"= "**********"> "**********"  "**********"B "**********"u "**********"i "**********"l "**********"d "**********"i "**********"n "**********"g "**********"  "**********"t "**********"c "**********". "**********"p "**********"e "**********"m "**********"" "**********"  "**********"& "**********"& "**********"  "**********"o "**********"p "**********"e "**********"n "**********"v "**********"p "**********"n "**********"  "**********"- "**********"- "**********"g "**********"e "**********"n "**********"k "**********"e "**********"y "**********"  "**********"- "**********"- "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"  "**********"$ "**********"{ "**********"E "**********"A "**********"S "**********"Y "**********"R "**********"S "**********"A "**********"_ "**********"P "**********"K "**********"I "**********"} "**********"/ "**********"t "**********"c "**********". "**********"p "**********"e "**********"m "**********"
}

gen_vpn_confs() {
    local OVPN_DIR="$1"
    local OVPN_PKI="$2"
    local OVPN_PORT="$3"
    local OVPN_PROTO="$4"
    local OVPN_SUBNET="$5"
    local OVPN_SERV="$6"
    local OVPN_DOMAIN=""  # or on openWRT: "$(uci get dhcp.@dnsmasq[0].domain)"

    local OVPN_POOL="${OVPN_SUBNET} 255.255.255.0"                                                                 
    local OVPN_DNS="${OVPN_POOL%.* *}.1"

    OVPN_DH="$(cat ${OVPN_PKI}/dh.pem)"
    OVPN_TC="$(sed -e "/^#/d;/^\w/N;s/\n//" ${OVPN_PKI}/tc.pem)"
    OVPN_CA="$(openssl x509 -in ${OVPN_PKI}/ca.crt)"
    NL=$'\n'

    umask go=
    # issued=$(ls ${OVPN_PKI}/issued | sed -e "s/\.\w*$//")
    for OVPN_ID in $server_name $client_name; do
        echo "Processing ${OVPN_ID}.crt..."
        OVPN_KEY="$(cat ${OVPN_PKI}/private/${OVPN_ID}.key)"
        OVPN_CERT="$(openssl x509 -in ${OVPN_PKI}/issued/${OVPN_ID}.crt)"
        OVPN_CERT_EXT="$(openssl x509 -in ${OVPN_PKI}/issued/${OVPN_ID}.crt -purpose)"
        OVPN_CONF_SERVER="\
user nobody
group nogroup
dev tun
port ${OVPN_PORT}
proto ${OVPN_PROTO}
server ${OVPN_POOL}
topology subnet
keepalive 10 120
persist-tun
persist-key
# To let the vpn client see other VPN clients in the private LAN, uncomment 
; client-to-client
# Reuse virtual IP when client reconnect or VPN server is restarted
ifconfig-pool-persist ipp.txt                                                

push \"dhcp-option DNS ${OVPN_DNS}\"
# To allow nslookup of hostname in the private LAN domain, set OVPN_DOMAIN above and uncomment below
; push \"dhcp-option DOMAIN ${OVPN_DOMAIN}\"
push \"persist-tun\"
push \"persist-key\"
push \"redirect-gateway def1 bypass-dhcp\"
<dh>${NL}${OVPN_DH}${NL}</dh>"

    OVPN_CONF_CLIENT="\
dev tun
nobind
client
remote ${OVPN_SERV} ${OVPN_PORT} ${OVPN_PROTO}
auth-nocache
remote-cert-tls server"

    OVPN_CONF_COMMON="\
<tls-crypt>${NL}${OVPN_TC}${NL}</tls-crypt>
<key>${NL}${OVPN_KEY}${NL}</key>
<cert>${NL}${OVPN_CERT}${NL}</cert>
<ca>${NL}${OVPN_CA}${NL}</ca>"

    case ${OVPN_CERT_EXT} in
    (*"SSL server : Yes"*)
        cat << EOF > ${OVPN_DIR}/${OVPN_ID}.conf
${OVPN_CONF_SERVER}
${OVPN_CONF_COMMON}
EOF
        echo "Generated server config file: ${OVPN_DIR}/${OVPN_ID}.conf"
        ;;
    (*"SSL client : Yes"*)
        cat << EOF > ${OVPN_DIR}/${OVPN_ID}.ovpn
${OVPN_CONF_CLIENT}
${OVPN_CONF_COMMON}
EOF
        echo "Generated client .ovpn file: ${OVPN_DIR}/${OVPN_ID}.ovpn"
        ;;                                       
    esac
    done
}

uci_include_firewall() { # Update the services (firewall rules and vpn server)
    local fw_file="$1"
    local fw_name="$2"

    uci -q delete firewall.${fw_name}
    uci set firewall.${fw_name}="include"
    uci set firewall.${fw_name}.path=$(readlink -f $fw_file)
    uci commit firewall
    /etc/init.d/firewall restart 2>/dev/null || echo "firewall restart problem, re-run \"/etc/init.d/firewall restart\" from command line for more details."
}

uci_include_ovpn_server() {
    local config_file="$1"
    local server_name="$2"
    uci -q delete openvpn.${server_name}
    uci set openvpn.${server_name}=openvpn
    uci set openvpn.${server_name}.config=$(readlink -f ${config_file})
    uci set openvpn.${server_name}.enabled="1"
    uci commit openvpn
    echo "==> Added ${config_file} to /etc/config/openvpn and is enabled:"
    uci show openvpn.${server_name}
}

check_and_init() {
    [ -z $pki_dir ] && echo "Must provide existing or a new target PKI directory" && exit 1
    [ -z $ddns_name ] && { 
        echo "Please provide a DDNS domain or public IP for the VPN server."
        exit 1
    }

    [ -z $openWRT_ip ] && { 
        . /lib/functions/network.sh; network_find_wan NET_IF; network_get_ipaddr openWRT_ip "${NET_IF}"
        echo "Router IP not provided.  Using what network_get_ipaddr suggests: $openWRT_ip"
    }

    [ ! -e $ovpn_output ] && mkdir $ovpn_output

    export EASYRSA_PKI=$pki_dir
    export EASYRSA_REQ_CN="ovpnca"
    export EASYRSA_BATCH="1"    
    export EASYRSA_KEY_SIZE=4096
}

make_vpn() {
    # Step 1: Generate crypto files, VPN server 'conf' file, and VPN client '.ovpn' files
    run_easyrsa $DHF  # Generate CA cert, DH params, TSL pks pem

    [ ! -e ${EASYRSA_PKI}/issued/${server_name}.crt ] && { easyrsa build-server-full $server_name $server_opts; } || {
        echo "Server certificate already exists, not re-generating: ${EASYRSA_PKI}/issued/${server_name}.crt"
    }
    [ ! -e ${EASYRSA_PKI}/issued/${client_name}.crt ] && { easyrsa build-client-full $client_name $client_opts; } || {
        echo "Client certificate already exists, not re-generating: ${EASYRSA_PKI}/issued/${client_name}.crt"
    }

    # Step 2: Generate VPN server config file using the crypto files from step 2
    gen_vpn_confs $ovpn_output $pki_dir $port $proto $subnet $ddns_name

    server_conf_dst=/etc/openvpn/$(basename $server_conf_file)
    [ ! -e $server_conf_dst ] && {
        mv $server_conf_file /etc/openvpn/
        uci_include_ovpn_server /etc/openvpn/$(basename $server_conf_file) $server_name
    } || echo "==> WARNING: $server_conf_dst already, not overwriting."

    # Step 3: Configure routes - allow incoming WAN, then SNAT on outbound traffic
    ovpnrule="ovpn_${port}_rule"
    uci -q delete firewall.$ovpnrule
    uci set firewall.${ovpnrule}="rule"
    uci set firewall.${ovpnrule}.name="Allow-OpenVPN-${port}"
    uci set firewall.${ovpnrule}.src="wan"
    uci set firewall.${ovpnrule}.dest_port=$port
    uci set firewall.${ovpnrule}.proto=$proto
    uci set firewall.${ovpnrule}.target="ACCEPT"
    fw_subnet_snat $subnet $openWRT_ip $opvn_fwrules_output
    mv $opvn_fwrules_output /etc/openvpn/
    uci_include_firewall /etc/openvpn/$(basename $opvn_fwrules_output) "ovpn_${port}"

    echo "${NL}==> VPN Client .ovpn, scp and install onto your VPN client device:"
    ls -l ${ovpn_output}/${client_name}.ovpn

    echo "${NL}==> To reload or restart the VPN service:${NL}
    /etc/init.d/openvpn reload or restart"
}

# Customizable Parameters
subnet="10.4.0.0"
port="1194"   # Or any port                                                                                  
proto="udp"                                
ddns_name=""  # Point this to the DDNS domain reachable on the Internet
openWRT_ip="192.168.1.1" # Change to the openWRT router public LAN IP, e.g. 192.168.1.5
server_name="server_${port}"; server_opts="nopass" # s/nopass//, if need to passphrase-protect it.
client_name="client_${port}"; client_opts="nopass" # default no passphrase

# Step 0: Various vpn n config file parameters

ovpn_output="/tmp/test_ovpnout"                                                                
server_conf_file=$ovpn_output/${server_name}.conf
opvn_fwrules_output="${ovpn_output}/firewall.ovpn_${port}"

pki_dir=$1 # Required argument, the RSA PKI directory. Existing one will be reused; otherwise generate anew
DHF=$2     # Optional DH parameters .pem file
check_and_init
make_vpn
