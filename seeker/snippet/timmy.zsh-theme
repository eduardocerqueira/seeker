#date: 2023-11-22T16:47:36Z
#url: https://api.github.com/gists/442a90440c5d0d6afbf298355cc40ce5
#owner: https://api.github.com/users/luigiinred

local ret_status="%(?:%{$fg_bold[green]%}➜ :%{$fg_bold[red]%}➜ %s)"
PROMPT=$'%{$fg[green]%}%n@%m: %{$reset_color%}%{$fg[blue]%}%~ %{$reset_color%}%{$fg_bold[blue]%}$(git_prompt_info)%{$fg_bold[blue]%} % %{$reset_color%}
${ret_status} %{$reset_color%} '
RPROMPT='%{$fg[gray]%}[%*]%{$reset_color%}'

PROMPT2="%{$fg_blod[black]%}%_> %{$reset_color%}"

ZSH_THEME_GIT_PROMPT_PREFIX="git:(%{$fg[red]%}"
ZSH_THEME_GIT_PROMPT_SUFFIX="%{$reset_color%}"
ZSH_THEME_GIT_PROMPT_DIRTY="%{$fg[blue]%}) %{$fg[yellow]%}✗%{$reset_color%}"
ZSH_THEME_GIT_PROMPT_CLEAN="%{$fg[blue]%})"
