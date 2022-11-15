#date: 2022-11-15T17:04:34Z
#url: https://api.github.com/gists/6a701b4449167ee39215096594a2e405
#owner: https://api.github.com/users/tswartz-vertax

# force `wp` commands to run as the `www-data` user
wp () {
    if [ $1 == "cli" ]; then
        sudo -H -u root bash -c "wp $* --allow-root"
    elif [ $(whoami) != "www-data" ]; then
        sudo -H -u www-data bash -c "wp $*"
    else
        command wp $*
    fi
}