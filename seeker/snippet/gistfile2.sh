#date: 2023-07-14T17:04:29Z
#url: https://api.github.com/gists/bfc24822931b55bcd750743f42f938a5
#owner: https://api.github.com/users/mattknox

# rpcsh -- Runs a function on a remote host
# This function pushes out a given set of variables and functions to
# another host via ssh, then runs a given function with optional arguments.
# Usage:
#   rpcsh -h remote_host [ -p ssh-port ] -u remote_login -v "variable list" \
#     -f "function list" -m mainfunc
#
# The "function list" is a list of shell functions to push to the remote host
# (including the main function to execute, and any functions that it calls)
# Use the "variable list" to send a group of variables to the remote host.
# Finally "mainfunc" is the name of the function (from "function list") 
# to execute on the remote side.  Any additional parameters specified gets
# passed along to mainfunc.

rpcsh() {
    if ! args=("$(getopt -l "rmthost:,rmthostport:,rmtlogin:,pushvars:,pushfuncs:,rmtmain:" -o "h:p:u:v:f:m:A" -- "$@")")
    then
        exit 1
    fi

    sshvars=( -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null )
    eval set -- "${args[@]}"
    while [ -n "$1" ]
    do
        case $1 in
            -h|--rmthost) rmthost=$2; shift; shift;;
            -p|--rmtport) sshvars=( "${sshvars[@]}" -p $2 ); shift; shift;;
            -u|--rmtlogin) rmtlogin=$2; shift; shift;;
            -v|--pushvars) pushvars=$2; shift; shift;;
            -f|--pushfuncs) pushfuncs=$2; shift; shift;;
            -m|--rmtmain) rmtmain=$2; shift; shift;;
            -A) sshvars=( "${sshvars[@]}" -A ); shift;;
            -i) sshvars=( "${sshvars[@]}" -i $2 ); shift; shift;;
            --) shift; break;;
        esac
    done
    rmtargs=( "$@" )

    ssh ${sshvars[@]} ${rmtlogin}@${rmthost} "
        $(declare -p rmtargs 2>/dev/null)
        $([ -n "$pushvars" ] && declare -p $pushvars 2>/dev/null)
        $(declare -f $pushfuncs 2>/dev/null)
        $rmtmain \"\${rmtargs[@]}\"
    "
}
