#date: 2021-11-05T17:04:05Z
#url: https://api.github.com/gists/6b8810da73e2fd331eaa5a5c2ffb148e
#owner: https://api.github.com/users/k-sriram

# Modified script from https://www.legroom.net/2010/05/05/generic-method-determine-linux-or-unix-distribution-name
# to include the os-release file as well. It might be more consistent, as lsb-release is now an installable package in most systems.
function find-os() {
    local UNAME OS
    # Determine OS platform
    UNAME=$(uname | tr "[:upper:]" "[:lower:]")
    # If Linux, try to determine specific distribution
    if [[ "$UNAME" == "linux" ]]; then
        # First check if os-release is available
        if [[ -f /etc/os-release ]]; then
            OS=$(grep -Po "(?<=^ID=).*(?=$)" /etc/os-release)
        # If available, use LSB to identify distribution
        elif [[ -f /etc/lsb-release || -d /etc/lsb-release.d ]]; then
            OS=$(lsb_release -i | cut -d: -f2 | sed s/'^\t'//)
        # Otherwise, use release info file
        else
            OS=$(ls -d /etc/[A-Za-z]*[_-][rv]e[lr]* | grep -v "lsb" | cut -d'/' -f3 | cut -d'-' -f1 | cut -d'_' -f1)
        fi
    fi
    # For everything else (or if above failed), just use generic identifier
    [[ "$OS" == "" ]] && OS=$UNAME
    OS=${(L)OS}
    echo -n $OS
}
# Values returned in various OSes
# Manjaro:manjaro
# CentOS:centos
# Ubuntu:ubuntu
# The following are not tested.
# Mint:linuxmint
# Fedora:fedora