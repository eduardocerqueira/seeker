#date: 2022-09-16T22:08:41Z
#url: https://api.github.com/gists/6b7835b241007633b4031da00c1e3ea4
#owner: https://api.github.com/users/emmanuelnk

function log_cmd()
{
    "$@"
    ret=$?

    if [[ $ret -eq 0 ]]
    then
        echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") user[ $USER ] status[ Success ] code[ $ret ] cmd[ $@ ]" >> bash_command.log
    else
        echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") user[ $USER ] status[ Error ] code[ $ret ] cmd[ $@ ]"  >> bash_command.log
        exit $ret
    fi
}

# example

log_cmd ls
# bash_command.log
# 2022-09-16T22:04:46Z user[ emmanuel ] status[ Success ] code[ 0 ] cmd[ ls ]

log_cmd lklj
# bash_command.log
# 2022-09-16T22:09:06Z user[ emmanuel ] status[ Error ] code[ 127 ] cmd[ lklj ]