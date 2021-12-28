#date: 2021-12-28T16:55:20Z
#url: https://api.github.com/gists/de7d939aa1a65aa8e0c4378028c4b27c
#owner: https://api.github.com/users/Amixp

#!/bin/sh

# ###############################
# Script to run PMM2 
# curl -fsSL https://gist.githubusercontent.com/askomorokhov/62ade5f05fe1c1cad0ae32664369d266/raw/d589ae2a24881cc8e52a9026aabec6e4a92f1f38/get-pmm.sh -o get-pmm2.sh ; chmod +x get-pmm2.sh ; ./get-pmm2.sh
#
#################################
set -o errexit
#set -o xtrace

root_is_needed='no'

gather_info() {
    GREEN='\033[0;33m'
    NC='\033[0m' # No Color
    printf "${GREEN}PMM Server Wizard Install${NC}%s\n"
    printf "%s\tPort Number to start PMM Server on (default: 443): "
    read listenPort
    : ${listenPort:="443"}
    printf "%s\tPMM Server Container Name (default: pmm-server): "
    read containerName
    : ${containerName:="pmm-server"}
    printf "%s\tOverride specific version (default: latest in 2.x series) format: 2.x.y: "
    read pmmVersion
    : ${pmmVersion:="2"}
    #printf "%s\n values: $listenPort $containerName $pmmVersion"
}

check_command() {
    command -v "$@" 1> /dev/null
}

run_root() {
    sh='sh -c'
    if [ "$(id -un)" != 'root' ]; then
        if check_command sudo; then
            sh='sudo -E sh -c'
        elif check_command su; then
            sh='su -c'
        else
            echo ERROR: root rights needed to run "$*" command
            exit 1
        fi
    fi
    ${sh} "$@"
}

install_docker() {
    if ! check_command docker; then
        printf "Installing docker"
        curl -fsSL get.docker.com -o /tmp/get-docker.sh \
            || wget -qO /tmp/get-docker.sh get.docker.com
        sh /tmp/get-docker.sh
        run_root 'service docker start' || :
    fi
    if ! docker ps 1> /dev/null; then
        root_is_needed='yes'
        if ! run_root 'docker ps 1> /dev/null' ; then
            echo ERROR: cannot run "docker ps" command
            exit 1
        fi
    fi
}

run_docker() {
    if [ "${root_is_needed}" = 'yes' ]; then
        run_root "docker $*"
    else
        sh -c "docker $*"
    fi
}


start_pmm() {
    run_docker "pull percona/pmm-server:$pmmVersion 1> /dev/null"

    if ! run_docker "inspect pmm-data 2> /dev/null 1> /dev/null" ; then
        run_docker "create -v /srv/ --name pmm-data percona/pmm-server:$pmmVersion /bin/true 1> /dev/null"
        printf "%sCreated PMM Data Volume: pmm-data%s\n"
    fi

    if run_docker "inspect pmm-server 2> /dev/null 1> /dev/null"; then
        pmmArchive="pmm-server-$(date "+%F-%H%M%S")"
        printf "%s\tExisting PMM Server found, renaming to $pmmArchive%s\n"
            run_docker 'stop pmm-server' || :
        run_docker "rename pmm-server $pmmArchive"
    fi
        runPMM="run -d -p $listenPort:443 --volumes-from pmm-data --name $containerName --restart always percona/pmm-server:$pmmVersion"

        run_docker "$runPMM 1> /dev/null"
        printf "Created PMM Server: $containerName%s\n"
        printf "  Use the following command if you ever need to update your container by hand:%s\n"
        printf "%s\t docker $runPMM\n"
}

show_message() {
    RED='\033[0;31m'
    LGREEN='\033[0;92m'
    NC='\033[0m' # No Color
    default_iface=$(awk '$2 == 00000000 { print $1 }' /proc/net/route)
    accessible_ip=$(ip addr show dev "$default_iface" | awk '$1 == "inet" { sub("/.*", "", $2); print $2 }')
    [[ $listenPort != "443" ]] && accessible_ip="$accessible_ip:$listenPort" || :
    printf "PMM Server has been successfully setup on this system!  You can access your new server using the following web address:%s\n"
    printf "%s\t${RED}https://$accessible_ip/${NC}%s\n"
    printf "The default username is '${LGREEN}admin${NC}' and password is '${LGREEN}admin${NC}' %s\n**Note** chrome may not trust the default SSL certificate on first load so type '${LGREEN}thisisunsafe${NC}' to bypass their warning%s\n"
}

main() {
    gather_info
    printf "Gathering/Downloading required components, this may take a moment %s\n"
    install_docker
    start_pmm
    show_message
}

main
exit 0
