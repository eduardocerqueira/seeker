#date: 2023-03-30T16:53:50Z
#url: https://api.github.com/gists/83873511281d5e7648f1bc06c50d278f
#owner: https://api.github.com/users/jan-swiecki

#!/bin/bash

# This script doesn't work with apt-get update (even with -q). Kill -0 exits immediately.

print_empty_lines() {
  for x in $(seq "$1"); do
    tput el
    echo
  done
}

stream_log_exec () {
  DISPLAY_LINES=10
  ERR_DISPLAY_LINES=20
  RED='\033[0;34m'
  RESET='\033[0m'

  logfile=$(mktemp --suffix .log)

  echo ">>> running: $*" >&2

  echo -en "\033[6n"; IFS=';' read -sdR -a pos; init_row="${pos[0]#*[}"; init_row=$((init_row-1))

  lines=$(tput lines)
  diff_lines=$((lines-init_row))
  if [[ $diff_lines -le "$DISPLAY_LINES" ]]; then
    init_row=$((lines - DISPLAY_LINES - 1))

    if [[ $init_row -lt 0 ]]; then
      echo "error: init_row<0 ($init_row)" >&2
      exit 1
    fi
  fi

  print_empty_lines "$DISPLAY_LINES"

  tput el1
  tput cup "$init_row" 0

  cleanup_pid () {
    tput cnorm
    if kill -0 "$1" &>/dev/null; then
      kill "$1"
    fi
  }

  goto_row () {
    tput el
    tput cup "${1:-$init_row}" 0
  }

  tail_stream () {
    goto_row "$init_row"
    print_empty_lines "$DISPLAY_LINES"
    goto_row "$init_row"
    cols=$(tput cols)
    echo -ne "$RED"
    sed_prefix=">> "
    sed_prefix_len="${#sed_prefix}"
    tail -n "${1:-$DISPLAY_LINES}" "$logfile" | fold "-w$((cols-sed_prefix_len))" | tail -n "${1:-$DISPLAY_LINES}" | sed -E 's/^/>> /g'
    echo -ne "$RESET"
  }

  # Function to handle command execution and dynamic output
  run_command() (
    # Execute the command and pipe the output to a temporary file
    "$@" > "$logfile" 2>&1 &

    # Get the process ID of the last background command
    local cmd_pid=$!

    tput civis
    trap "cleanup_pid '$cmd_pid'" EXIT

    # echo -ne "$RED"

    tail_stream
    # Continuously display the last N lines of the output
    while kill -0 $cmd_pid 2>/dev/null; do
      tail_stream
      sleep 0.1
    done
    tail_stream

    num_output_lines=$(<"$logfile" wc -l)

    if [[ "$num_output_lines" -eq 0 ]]; then
      echo ">> (No output)" >&2
    fi

    echo -en "\033[6n"; IFS=';' read -sdR -a pos; curr_row="${pos[0]#*[}"; #curr_row=$((curr_row-1))

    if [[ $num_output_lines -lt "$DISPLAY_LINES" ]]; then
      go_up_by=$((num_output_lines+1))
    else
      go_up_by=$((DISPLAY_LINES+1))
    fi


    set +eo pipefail
    wait "$cmd_pid"
    exit_code=$?

    goto_row "$((curr_row-go_up_by))"
    if [[ $exit_code -gt 0 ]]; then
      tail_stream "$ERR_DISPLAY_LINES"
    else
      print_empty_lines "$DISPLAY_LINES"
      goto_row "$((curr_row-go_up_by-1))"
    fi

    # echo -ne "$RESET"
    if [[ $exit_code -gt 0 ]]; then
      # echo ">>> running: $* (command exited with status $exit_code, full log available at: $logfile))" >&2
      echo ">>> command exited with status $exit_code (full log available at: $logfile)" >&2
      return "$exit_code"
    else
      echo ">>> success: $* (full log available at: $logfile)" >&2
      # echo ">>> command finished successfully (full log available at: $logfile)" >&2
    fi
  )


  run_command "$@"
}

stream_log_exec "$@"