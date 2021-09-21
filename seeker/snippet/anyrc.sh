#date: 2021-09-21T16:52:31Z
#url: https://api.github.com/gists/b5c1d4ef9fb7d3196cdeb2e2785a6124
#owner: https://api.github.com/users/degzcs

#                                                                                                                                                                 
# Vim                                                                                                                                                              
#

bindkey -v # User vim commands on the terminal
# Better searching in command mode
bindkey -M vicmd '?' history-incremental-search-backward
bindkey -M vicmd '/' history-incremental-search-forward

# Beginning search with arrow keys
bindkey "^[OA" up-line-or-beginning-search
bindkey "^[OB" down-line-or-beginning-search
bindkey -M vicmd "k" up-line-or-beginning-search
bindkey -M vicmd "j" down-line-or-beginning-search

#
# Vim
#

stty -ixon