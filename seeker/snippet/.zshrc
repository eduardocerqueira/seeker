#date: 2022-12-07T17:05:10Z
#url: https://api.github.com/gists/28a20373161e3bc8bfa9a903c3fab32f
#owner: https://api.github.com/users/cleaver

# adapted from: https://www.growingwiththeweb.com/2018/01/slow-nvm-init.html

if [ -s "$HOME/.nvm/nvm.sh" ] && [ ! "$(whence -w __init_nvm)" = "__init_nvm: function" ]; then
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/bash_completion" ] && . "$NVM_DIR/bash_completion"
  declare -a __node_commands=('nvm' 'node' 'npm' 'yarn' 'gulp' 'grunt' 'webpack')
  function __init_nvm() {
    for i in "${__node_commands[@]}"; do unalias $i; done
    . "$NVM_DIR"/nvm.sh
    unset __node_commands
    unset -f __init_nvm
  }
  for i in "${__node_commands[@]}"; do alias $i='__init_nvm && '$i; done
fi