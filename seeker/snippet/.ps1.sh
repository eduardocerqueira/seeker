#date: 2024-04-09T16:51:27Z
#url: https://api.github.com/gists/ccbeb2a64f3b5b574682b043583b9c83
#owner: https://api.github.com/users/miro662

RESET_COLOR="\033[0m"
DIR_COLOR="\033[1m\033[34m" # blue
HOME_COLOR="\033[1m\033[32m" # green
COMMAND_COLOR="\033[2m" # faint
PROMPT="$"

function workdir {
    WORK_DIR=`pwd`

    HOME_PATTERN=${HOME//"/"/"\/"}
    WORK_DIR=${WORK_DIR/$HOME_PATTERN/"$HOME_COLOR~"}
    echo -e -n "$DIR_COLOR$WORK_DIR"
    echo -e -n "$RESET_COLOR "
}

function prompt {
	echo -e -n "$PROMPT $RESET_COLOR$COMMAND_COLOR"
}

function prompt_command {
    workdir
    prompt
}

PS0="$RESET_COLOR"
PS1='$(prompt_command)'
