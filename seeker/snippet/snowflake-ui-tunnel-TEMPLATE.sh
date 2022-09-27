#date: 2022-09-27T17:27:32Z
#url: https://api.github.com/gists/142d3ac6885656e2008a96bc68c8e2c5
#owner: https://api.github.com/users/nturdo-cedar

#!/bin/bash
shopt -s nocasematch

###############
# Setups a tunnel to connect to snowflakes private link through
# jump.cedar.com.  Addes the necessary /etc/hosts entries to allow
# the hosts browser to connect directly to snowflakes ui
#
# Args:
#   - desired tunnel state. acceptable options "up" | "down" | "status"
#
# In "up" mode this will:
#   - Add new IP address to the loopback interface for ssh to bind to
#   - Add entries for snowflakes privatelink hosts to your /etc/hosts
#   - setup ssh tunnel through jump to forward traffic to snowflake
#
# In "down" mode this will:
#   - Kill any open tunnels
#   - Remove hosts file entries
#   - Remove added loopback alias addresses
#
# "status" check if any of the pieces is borked. Recommended action is to
# run down then up
###############

###############
# Client Specific Values
###############

SF_PRIVATELINK="<URL>"
SF_PRIVATELINK_OCSP="ocsp.<URL>"

###############
# Static Globals
###############

JUMP_HOST="jump.cedar.com"
LOOPBACK_ALIAS="127.0.1.1"
LOOPBACK_SUBNET="255.0.0.1"
HOSTS_FILE_MARKER="SNOWFLAKEUITUNNELMARK"
SSH_USER=$USER
SSH_OPTS="-f -N -n -M"
TUNNEL_CONTROL=${HOME}/.ssh/snowflake_tunnel.ctl
VERSION="2021.11.08"
USAGE="""
Usage: command [-h|--help] [-v|--version] DesiredState
Options:
    -h, --help                 Prints out usage message.
    -V, --version              Prints out version Number.
Argument:
    DesiredState               Operation you wish to perform in regards to the snowflake tunnel.
                               Valid Values: up, down, status
Prerequisites:
    Execute the script while you have SUDO permissions.
    Enable access to JumpServers by checking in your SSH key to MonoRepo.
"""

###############
# Helpers
###############

add_hosts_entry () {
    rm_hosts_entry
    printf "%s\t%s %s\t#%s\n" $LOOPBACK_ALIAS $SF_PRIVATELINK $SF_PRIVATELINK_OCSP $HOSTS_FILE_MARKER | sudo tee -a /etc/hosts >/dev/null 2>&1
}

rm_hosts_entry () {
    cp /etc/hosts /tmp/hosts.bak-$(date +%s)
    sudo sed -i -e "/${HOSTS_FILE_MARKER}/d" /etc/hosts
}

check_hosts_entry () {
    grep $HOSTS_FILE_MARKER /etc/hosts >/dev/null 2>&1
}

add_interface () {
    #sudo ifconfig lo0 alias $LOOPBACK_ALIAS $LOOPBACK_SUBNET >/dev/null 2>&1
    sudo ifconfig lo0 alias $LOOPBACK_ALIAS $LOOPBACK_SUBNET
}

rm_interface () {
    sudo ifconfig lo0 -alias $LOOPBACK_ALIAS >/dev/null 2>&1
}

check_interface () {
    ifconfig | grep "inet ${LOOPBACK_ALIAS}" >/dev/null 2>&1
}

create_tunnel () {
    sudo -E ssh $SSH_OPTS -S $TUNNEL_CONTROL -L ${LOOPBACK_ALIAS}:443:${SF_PRIVATELINK}:443 -L ${LOOPBACK_ALIAS}:80:${SF_PRIVATELINK_OCSP}:80 ${SSH_USER}@${JUMP_HOST}
}

kill_tunnel () {
    sudo ssh -S $TUNNEL_CONTROL -O exit $JUMP_HOST >/dev/null 2>&1
}

check_tunnel () {
    sudo ssh -S $TUNNEL_CONTROL -O check $JUMP_HOST 2>&1 | grep "Master running" >/dev/null 2>&1
}

check_everything () {
    check_hosts_entry && check_interface && check_tunnel
}

validate_root_access(){
    sudo -v >/dev/null 2>&1
}

###############
# Main
###############



# Command Line Options Processing
ARGUMENTS=()
for arg in $@; do
    case ${arg} in
        -v|--version)
            echo "${VERSION}"
            exit 0
        ;;
        -h|--help)
            echo "${USAGE}"
            exit 0
        ;;
        -s| --silent)
            SILENT=true
            shift
        ;;
        UP)
            ARGUMENTS+=("$1")
            shift
        ;;
        DOWN)
            ARGUMENTS+=("$1")
            shift
        ;;
        STATUS)
            ARGUMENTS+=("$1")
            shift
        ;;
        *)
            echo "Unsupported argument/s provided. Please review usage below."
            echo "$USAGE"
            exit 1
    esac
done

if [[ "${#ARGUMENTS[@]}" > 1 ]]; then
    echo "Too many arguments provided. Please review usage below."
    echo "$USAGE"
    exit 1
elif [[ "${#ARGUMENTS[@]}" = 0 ]]; then
    echo "No arguments provided. Please review usage below."
    echo "$USAGE"
    exit 1
else
    STATE=${ARGUMENTS[0]}
fi
# End Command Line Options Processing 

# Validate sudo access
validate_root_access
if [[ $? != 0 ]]; then
    echo "Script requires root (sudo) permissions. Please review usage below."
    exit 1
fi
# End Validate sudo access

# Perform state operation
case $STATE in
    UP)
        if check_everything; then
            echo "Tunnel already up..."
            exit 0
        fi
        echo "Requesting sudo on your computer. Please enter your computer password when prompted for 'Password: "**********"
        add_interface
        if [[ $? != 0 ]]; then
            echo "Something went wrong during 'add_interface'"
            exit 1
        fi
        add_hosts_entry
        if [[ $? != 0 ]]; then
            echo "Something went wrong during 'add_hosts_entry'"
            exit 1
        fi
        echo "Requesting your Jump Server 2FA code. Please enter your 2FA code when prompted for 'Verification code:'"
        create_tunnel
        if [[ $? != 0 ]]; then
            echo "Something went wrong during 'create_tunnel'"
            exit 1
        fi
        echo "You can now load: https://${SF_PRIVATELINK}"
    ;;
    DOWN)
        echo "Requesting sudo on your computer. Please enter your computer password when prompted for 'Password: "**********"
        kill_tunnel
        rm_interface
        rm_hosts_entry
    ;;
    STATUS)
        if check_everything; then
            echo "You look good to go..."
        elif ! check_hosts_entry || ! check_interface || ! check_tunnel; then
            echo "Missing one or more components. Run down then up to fix..."
        else
            echo "Tunnel is off"
        fi
    ;;
esac
# End Perform state operaiton
