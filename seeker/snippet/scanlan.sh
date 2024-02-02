#date: 2024-02-02T16:53:52Z
#url: https://api.github.com/gists/944ee5194eb1526d125aff5d06eb0cb8
#owner: https://api.github.com/users/FlyingFathead

#!/bin/bash
#
# 	>>> SCANLAN <<<
#
# this script can used for i.e. checking out what's on your LAN
# requires `nmap`, `xmlstarlet` and `lolcat` (just because)
# adjust to your own ip range as needed. 
# NO WARRANTIES, use only for your own LAN diagnostics and at your own risk
#
#
## INFO
#
scanlan_ver='v0.8 -- 02 Feb. 2024		(c) 2021-2024 FlyingFathead'
#
## viivo
#
function viibla() {
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' - ;
}
function viivo() {
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' - | lolcat -f;
}
#
## ekko
#
function ekho() {
echo -e "::: \e[0m$1"
}
function ekko() {
echo -e "::: \e[1m$1\e[0m"
}
function ekk0() {
echo -e "\e[1m$1\e[0m" | lolcat -f ;
}
## Colors
# red
prn_pun="\e[31m"
# green
prn_vih="\e[32m"
# yellow
prn_kel="\e[93m"
# white
prn_val="\e[97m"
##
highlight() { grep --color=always -e "^" -e "$*" ; }
##
function list_local_ips() { 
    echo "Local Network Interfaces and IP Addresses:" &&
    viivo &&
    # Use `ip` command to list IPs for all interfaces
    ip -br address | awk '$1 != "lo" {print $1, $3}' | while read -r interface ip_address; do
        echo -e "Interface: \e[1m$interface\e[0m, IP Address: $ip_address"
    done
    viivo &&
    echo ""
}
#
## Modify your nmap command to include XML output
nmap_xml_output="$scanlanlog_dir/nmap_scan_$(date +"%Y_%m_%d___%H_%M-%S").xml"
#
## Scan the LAN
#
function scanlan() {
	echo "" && echo "" &&
	viivo &&
	ekk0 "::: SCANLAN - $scanlan_ver" &&
	viivo &&
	##
	ekko "This tool must be run as 'sudo'. If prompted, enter your sudo password." &&
	viibla &&
	sudo echo "" &&
	echo "" &&
	##
	scanlanlog_dir="$HOME/.logs/.scanlan"
	scanlanlog_file="$(date +"%Y_%m_%d___%H_%M-%S").log"

	export scanlanlog_dir
	export scanlanlog_file

	if [ ! -d "$scanlanlog_dir" ]; then
		viibla &&
		ekko "scanlanlog_dir -- directory doesn't exist, creating it." &&
		mkdir -p "$scanlanlog_dir" &&
		viibla
	fi
	##
	if [ ! -d "$scanlanlog_dir" ]; then
		viibla &&
		ekko "$prn_pun" '[ERROR!]' "$scanlanlog_dir -- unable to create directory!" &&
		ekko "$prn_pun" '[ERROR!]' "Exiting!" &&
		viibla &&
		exit 0 
	fi
	##
	## file-friendly date = $(date +"%Y_%m_%d___%H_%M-%S").log
	export scanlanlog_file
	# notes ...
	## export output to log _with_ ANSI codes ===>>>
	## exec > >(tee -i "$scanlanlog_dir/$scanlanlog_file")
	## export output to log _without_ ANSI codes ===>>>
	##
	## logging ...
	##
	exec > >( tee >( sed 's/\x1B\[[0-9;]*[JKmsu]//g' >> "$scanlanlog_dir/$scanlanlog_file" ) )
	exec 2>&1
	##
	viibla &&
	ekho "Log started into: $scanlanlog_dir/$scanlanlog_file"
	viibla &&
	export defugateway0=$(/sbin/ip route | awk '/default/ { print $3 }')
	export defugateway1=$(echo $defugateway0 | tr '\n' ' ')
	read ip_a ip_b ip_c ip_d ip_e <<<"${defugateway1// / }"
	##
	if [ ! -z $ip_e ]; then
		ekko "$prn_pun""[HOLY SMOKES!] $prn_val""$ip_e is where we draw the line." &&
		ekko "You've got issues or more than enough default gateways.";
	fi
	echo "$ip_a" | grep select
	export ip_s=$(echo $ip_a \')
	##
	ekko "Default gateway(s) seem(s) to be: \
	$(if [ ! -z $ip_a ]; then echo $ip_a; fi) \
	$(if [ ! -z $ip_b ]; then echo -e "$prn_kel""AND $prn_val""$ip_b"; fi) \
	$(if [ ! -z $ip_c ]; then echo -e "$prn_pun""AND $prn_val""$ip_c"; fi) \
	$(if [ ! -z $ip_d ]; then echo -e "$prn_pun""AND !" "$prn_val""$ip_d"; fi)" &&
	viibla &&
	echo "" &&
	ekko "Results from 'ip route':" &&
	ip route &&
	echo "" &&
	viibla &&
	ekko "ARP check, from 'arp':" &&
	viibla &&
	arp | highlight "${ip_s::-1}"
	## viivo &&
	echo "" &&
	ekko "Default gateway ($ip_a) is highlighted in $prn_pun""red\e[0m." &&
	viibla &&
	echo "(NOTE: The table shows the IP addresses in the left column, and MAC addresses in the middle. If the table contains two different IP addresses that share the same MAC address, then you are probably undergoing an ARP poisoning attack [unless you are using very special types of routings or setups, such as multiple NIC's etc.])" &&
	echo "" &&
	viibla &&
	ekko "Scanning the LAN with: 'sudo nmap -sS -O -v 192.168.100.0/24'" &&
	viibla &&

	sudo nmap -sS -O -v -oX "$nmap_xml_output" 192.168.100.0/24

	echo "" &&
	echo "" &&
	viivo &&
	ekko "SCANLAN finished at: $(date)"
	ekko "SCANLAN results log at: $scanlanlog_dir/$scanlanlog_file" 
	viivo &&

	# process the XML output to list active systems and their details
	if command -v xmlstarlet >/dev/null; then
	    echo "::: Active Systems and Details:"
	    viivo &&
	    xmlstarlet sel -t -m "//host[status/@state='up']" \
		-v "address[@addrtype='ipv4']/@addr" -o " " \
		-v "address[@addrtype='mac']/@addr" -o " (" \
		-v "address[@addrtype='mac']/@vendor" -o ")" -n \
		-m "ports/port" -o "  Open Port: " -v "@portid" \
		-o " (" -v "state/@state" -o ", " -v "service/@name" -o ")" -n \
		-m "os/osmatch" -o "  OS Estimate: " -v "@name" -n \
		"$nmap_xml_output" | while read -r line; do
		    if [[ $line =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
		        echo -e "\e[1mIP: $line\e[0m"  # Highlight IP in bold
		    else
		        echo "$line"  # Normal output for other lines
		    fi
		done
	else
	    echo "xmlstarlet is not installed. Can't process the XML output. You can install xmlstarlet with: `sudo apt-get install xmlstarlet`"
	fi

	## 
	# call the function to list local IPs
	list_local_ips
	viivo &&
	## done.

	## query to read the scan log
		while true
		do
		 read -r -p "Read the log now (answering [n]o will quit)? [Y/n] " input
		 
		 case $input in
		     [yY][eE][sS]|[yY])
		 less "$scanlanlog_dir/$scanlanlog_file" &&
		 echo "" &&
		 viivo &&
		 echo ""
		 break
		 ;;
		     [nN][oO]|[nN])
		 exit 0
		 break
			;;
		     *)
		 echo "Invalid input!"
		 ;;
		 esac
		done


	echo ""
}

scanlan

viivo &&
ekko "::: SCANLAN results log at: $scanlanlog_dir/$scanlanlog_file" 
viivo &&

function info() {
printf "

More reading at:

https://nmap.org/book/osdetect-usage.html -- Usage.

"
viivo

}
