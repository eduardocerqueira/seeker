#date: 2023-01-30T17:10:02Z
#url: https://api.github.com/gists/7116c5a0d7611655c980ef990f3a14c7
#owner: https://api.github.com/users/mpounsett

if [[ $OSTYPE =~ 'darwin.*' ]]; then
    # Inspired by https://gist.github.com/bashbunni/f6b04fc4703903a71ce9f70c58345106
    # requires:
    # - https://github.com/caarlos0/timer
    # - https://github.com/julienXX/terminal-notifier
    #
    POMO_WORKMSG="Work timer is up! Take a break."
    POMO_WORKTIME=60m
    POMO_RESTMSG="Break is over!  Time to get back to work."
    POMO_RESTTIME=10m
    POMO_SOUND=Blow
    POMO_OPTIONS=(-title Pomodoro -sound ${POMO_SOUND})
    POMO_DATE=(date "+%H:%M")
    alias work="timer ${POMO_WORKTIME} && terminal-notifier ${POMO_OPTIONS}\
        -message '\[$(${POMO_DATE})] ${POMO_WORKMSG}'"
    alias rest="timer ${POMO_RESTTIME} && terminal-notifier ${POMO_OPTIONS}\
        -message '\[$(${POMO_DATE})] ${POMO_RESTMSG}'"
fi
