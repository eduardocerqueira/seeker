#date: 2024-11-19T16:56:33Z
#url: https://api.github.com/gists/4f249c47d75bd9d7fa9745c5bed8f15e
#owner: https://api.github.com/users/matteopessina

# Terminal
export PROMPT_COMMAND='PS1_CMD1=$(git branch --show-current 2>/dev/null)'
export PS1='\[\033]0;${PWD/#$HOME/\~}\007\]\[\033[32m\]\u@\h \[\033[35m\]\[\033[33m\]\W \[\033[36m\](${PS1_CMD1}) \[\033[0m\]$ '

# Kubernetes
source <(kubectl completion bash)
alias k=kubectl
complete -F __start_kubectl k
export KUBE_EDITOR=vim
