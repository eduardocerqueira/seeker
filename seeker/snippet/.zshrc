#date: 2022-01-27T17:00:04Z
#url: https://api.github.com/gists/18017d12c3812650330e06b20a6c06b6
#owner: https://api.github.com/users/Roytangrb

# Adapted from: https://frantic.im/notify-on-completion/

# rmb to chmod +x ~/notifyme.scpt
alias notifyme="~/notifyme.scpt"

function f_notifyme {
  LAST_EXIT_CODE=$?
  CMD=$(fc -ln -1)
  notifyme "$CMD" "$LAST_EXIT_CODE" &
}

export PS1='$(f_notifyme)'$PS1