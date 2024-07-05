#date: 2024-07-05T16:44:51Z
#url: https://api.github.com/gists/20c5223f6713d52218d8ef4b11ddaf70
#owner: https://api.github.com/users/ika-musuko

# PROMPT="[%*] %n:%c $(git_prompt_info)%(!.#.$) "
PROMPT='[%*] %{%{$fg[green]%}%~%{$reset_color%}$(git_prompt_info) %(!.#.$) '

ZSH_THEME_GIT_PROMPT_PREFIX=" %{$fg[yellow]%}("
ZSH_THEME_GIT_PROMPT_SUFFIX=")%{$reset_color%}"
