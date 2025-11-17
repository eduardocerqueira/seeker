#date: 2025-11-17T16:43:00Z
#url: https://api.github.com/gists/a9d9a4c7aacb19ba2d972e133b9206b7
#owner: https://api.github.com/users/jdavidrcamacho

# Command logger for Fluentd 
if [ -n "$PS1" ] && [ -z "$BASH_COMMAND_LOGGER_SET" ]; then 
  export BASH_COMMAND_LOGGER_SET=1 
  shopt -s histappend 
  export HISTTIMEFORMAT="%F %T " 

  LOG_FILE="/var/log/test.log"
  LOG_HOST="$(hostname)"
  #IP address on the machine
  LOG_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

  PROMPT_COMMAND='
    LAST_CMD=$(HISTTIMEFORMAT= history 1 | sed "s/^ *[0-9]\+ *//");
    printf "%s host=%q ip=%q user=%q tty=%q pwd=%q cmd=%q\n" \
      "$(date --iso-8601=seconds)" "$LOG_HOST" "$LOG_IP" "$USER" "$(tty 2>/dev/null)" "$PWD" "$LAST_CMD" >> "$LOG_FILE";
    history -a
  '
fi
