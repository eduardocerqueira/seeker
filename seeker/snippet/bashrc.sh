#date: 2022-06-16T17:06:11Z
#url: https://api.github.com/gists/6ae3a76ae2b95b9c248ca008ab1e9884
#owner: https://api.github.com/users/raulnegreiros

# notifies when a command finishes its execution
# a desktop notification is sent with the execution time and result also an audio is played
# a successful result 
# usage: n <command>
# example: n sleep 10
# example: n ls ~/invalidFile
#
# depends on: bash4, notify-send, ffplay
n() {
    # custom your preferences 
    declare -A ___CONFIG_SUCCESS=(
        ["sound"]="/usr/share/sounds/gnome/default/alerts/bark.ogg"
        ["text"]="SUCCESS"
        ["icon"]="emblem-default" # it can be found in: /usr/share/icons/gnome/32x32
    )
    declare -A ___CONFIG_FAILURE=(
        ["sound"]="/usr/share/sounds/gnome/default/alerts/drip.ogg"
        ["text"]="FAILED"
        ["icon"]="emblem-important" # it can be found in: /usr/share/icons/gnome/32x32
    )

    declare -A ___CONFIG

    # execute and track the time
    ___START=$(date +%s)
    "$@"
    ___RESULT=$?
    ___END=$(date +%s)

    # load the configuration depending on result
    if [ "${___RESULT}" == "0" ] ; then # success
        for ___KEY in "${!___CONFIG_SUCCESS[@]}"; do
            ___CONFIG[${___KEY}]=${___CONFIG_SUCCESS[${___KEY}]}
        done
    else # failure
        for ___KEY in "${!___CONFIG_FAILURE[@]}"; do
            ___CONFIG[${___KEY}]=${___CONFIG_FAILURE[${___KEY}]}
        done
    fi

    # send notification
    notify-send -i ${___CONFIG["icon"]} "${___CONFIG["text"]}" "execution time: $((${___END}-${___START}))s"
    ffplay -nodisp -autoexit -loglevel quiet -nostats ${___CONFIG["sound"]}

    # clean up 
    unset ___START ___RESULT ___END ___CONFIG ___CONFIG_SUCCESS ___CONFIG_FAILURE ___KEY
}