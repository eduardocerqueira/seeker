#date: 2025-10-22T17:13:35Z
#url: https://api.github.com/gists/6e772368b105056df0cfcbd2b0055ac0
#owner: https://api.github.com/users/Astroxslurg

GREEN_BOLD='\[\033[01;32m\]'
BLUE_BOLD='\[\033[01;34m\]'
YELLOW='\[\033[0;33m\]'
CYAN='\[\033[0;36m\]'
RESET='\[\033[00m\]'
K8S_SYMBOL=$'\xE2\x98\xB8'

export PS1_USER=true
export PS1_PATH=true
export PS1_K8S=true
export PS1_GIT=true

get_prompt () {
  PS1='${debian_chroot:+($debian_chroot)}'

  if [ "$PS1_USER" = true ]; then
    PS1+="$GREEN_BOLD\u@\h$RESET:"
  fi

  if [ "$PS1_PATH" = true ]; then
    PS1+="$BLUE_BOLD\w$RESET"
  else
    PS1+="$BLUE_BOLD\W$RESET"
  fi

  if [ "$PS1_K8S" = true ]; then
    PS1+=" $CYAN$K8S_SYMBOL "
    PS1+='$(kubectl config view --minify -o jsonpath="{..namespace}")'
  fi

  if [ "$PS1_GIT" = true ]; then
    PS1+="$YELLOW"
    PS1+='$(__git_ps1)'
  fi

  PS1+="$RESET"
  PS1+='\$ '
}

export PROMPT_COMMAND=get_prompt
