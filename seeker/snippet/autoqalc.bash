#date: 2023-07-14T16:45:47Z
#url: https://api.github.com/gists/6f46bff89c444ec84abb9a4d449c0e42
#owner: https://api.github.com/users/jkcdarunday

# Author: Jan Keith Darunday <github@jkcdarunday.mozmail.com>
# Description: A script that forwards the input to qalc when the not-found command starts with a number
# Purpose: So you can directly execute mathematical expressions and currency conversions in the terminal
# Usage: Source the file in your bashrc/zshrc (i.e. `source ~/.zsh/autoqalc.bash`)

if typeset -f command_not_found_handle > /dev/null; then
    eval original_"$(declare -f command_not_found_handle)"
else
    original_command_not_found_handle() {
        "$1: command not found"
    }
fi


command_not_found_handle() {
    if [[ $1 =~ ^[0-9] ]]; then
        echo "autoqalc: command not found, using qalc instead since input starts with a number..."
        qalc "$@"
    else
        original_command_not_found_handle "$@"
    fi
}

