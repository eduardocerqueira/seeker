#date: 2023-03-06T17:03:39Z
#url: https://api.github.com/gists/df48ebd0d19c4bd2aef6d72e1111b49b
#owner: https://api.github.com/users/LuisPalacios

#!/bin/bash
#
## Servidor ‘sur’ 
##
## Este fichero está relacionado con este apunte: 
## https://www.luispa.com/linux/2014/10/19/bridge-ethernet.html
## 
#
# Quita las iptables por completo, lo permite todo.
#

# Averiguo nombres de las interfaces
. /root/firewall/sur_firewall_inames.sh

# Funciones
set_table_policy() {
    local chains table=$1 policy=$2
    case ${table} in
        nat)    chains="PREROUTING POSTROUTING OUTPUT";;
        mangle) chains="PREROUTING INPUT FORWARD OUTPUT POSTROUTING";;
        filter) chains="INPUT FORWARD OUTPUT";;
        *)      chains="";;
    esac
    local chain
    for chain in ${chains} ; do
        iptables -t ${table} -P ${chain} ${policy}
        #echo "iptables -t ${table} -P ${chain} ${policy}"
    done
}


# Limpiar las tablas de routing
#
export iptables_proc="/proc/net/ip_tables_names"
for a in $(cat ${iptables_proc}) ; do

    set_table_policy $a ACCEPT

   iptables -F -t $a
   iptables -X -t $a
done

# Limpiar iptables por completo
iptables -P INPUT ACCEPT
iptables -P FORWARD ACCEPT
iptables -P OUTPUT ACCEPT
iptables -t nat -F
iptables -t mangle -F
iptables -F
iptables -X