#date: 2024-01-04T17:08:41Z
#url: https://api.github.com/gists/10222c841fe5d3d4ec0f9fb0ad060043
#owner: https://api.github.com/users/mabster314

shopt -s promptvars

function nonzero_return() {
    RETVAL=$?
    [ $RETVAL -ne 0 ] && echo "$RETVAL "
}
LBLUE='\[\033[01;32m\]'
LILAC='\[\033[01;34m\]'
LRED='\[\033[00;91m\]'
PS_CLEAR='\[\033[0m\]'
PS1="${LBLUE}\u@\h:${PS_CLEAR}${LILAC}\w${PS_CLEAR} ${LRED}\$(nonzero_return)${PS_CLEAR}\$ "