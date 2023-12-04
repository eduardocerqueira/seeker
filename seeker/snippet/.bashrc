#date: 2023-12-04T16:59:27Z
#url: https://api.github.com/gists/5a3634efb5811a5c2bd3e5162a7dd4ab
#owner: https://api.github.com/users/marcelsyben

# Generated with bashrc generator: https://alexbaranowski.github.io/bash-rc-generator/
# History Settings

export HISTFILESIZE=2000
export HISTSIZE=2000
export HISTIGNORE="cd*:false:history:htop:ls*:ll*:la:l:popd:pushd*:reset:top:true"
export HISTCONTROL="erasedups"
export HISTTIMEFORMAT="%Y-%m-%d %T "
export HISTFILE="~/.bash_history"
shopt -s histappend
export PROMPT_COMMAND="history -a; history -c; history -r; $PROMPT_COMMAND"

# Aliases
alias cd="pushd"
alias back="popd"
popd()
{
  builtin popd > /dev/null
}
pushd()
{
  if [ $# -eq 0 ]; then
    builtin pushd "${HOME}" > /dev/null
  elif [ $1 == "-" ]; then
      builtin popd > /dev/null
  else
    builtin pushd "$1" > /dev/null
  fi
}
alias ..="cd .."
alias ...="cd ../../../"
alias ....="cd ../../../../"
alias .....="cd ../../../../"
alias .....="cd ../../../../"
alias ls='ls -oAh --color=auto'
alias rm='rm -rmv --preserve-root'
alias mv='mv -v'
alias cp='cp -v'
alias mkdir="mkdir -pv"
alias vi="vim"
alias grep='grep --color=auto'
alias egrep='egrep --color=auto'
alias fgrep='fgrep --color=auto'
hash colordiff &> /dev/null && alias diff='colordiff'
alias pbcopy="xclip -selection c"
alias pbpaste="xclip -selection clipboard -o"
alias now='date +"%F-%T; %V week"'
alias my_ip='curl -s ifconfig.co/json | python3 -m json.tool'
# Extra options

export EDITOR="nano"
export VISUAL="nano"
export PAGER="less"
shopt -s checkwinsize
export IGNOREEOF=2