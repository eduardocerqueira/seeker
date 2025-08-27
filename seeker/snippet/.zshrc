#date: 2025-08-27T17:01:29Z
#url: https://api.github.com/gists/7bb722122cedb199af5de319bd1e80b4
#owner: https://api.github.com/users/dannegm

#!/bin/env bash
# SHELL/ZSH Setup

source ~/.functions

# Enable add-zsh hook
autoload -U add-zsh-hook

# Autoload NVM version
add-zsh-hook chpwd load_nvm_version
load_nvm_version