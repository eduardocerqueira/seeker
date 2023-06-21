#date: 2023-06-21T17:09:00Z
#url: https://api.github.com/gists/65f53cf0c773e2cfd11c738b314e6d72
#owner: https://api.github.com/users/danielcarr

# Enable completion
autoload -Uz compinit && compinit

# Required for git completion
zstyle ':completion:*:*:git:*' script ~/.bashrc.d/git-completion.bash
fpath=(~/.zshrc.d $fpath)

for config in "${HOME}"/.zshrc.d/*.zsh; do
    . "${config}"
done
unset -v config
