#date: 2023-04-03T16:46:11Z
#url: https://api.github.com/gists/479e591dd9b2ec83bad612e11fbf04ba
#owner: https://api.github.com/users/leftl

# see: https://github.com/marlonrichert/zsh-snap/tree/main
if [[ ! -f ${ZDOTDIR}/plugins/zsh-snap/znap.zsh ]]; then
    command git clone --depth 1 https://github.com/marlonrichert/zsh-snap.git ${ZDOTDIR}/plugins/zsh-snap
fi
zstyle ':znap:*' repos-dir ${ZDOTDIR}/plugins

source ${ZDOTDIR}/plugins/zsh-snap/znap.zsh

# `znap source` automatically downloads and starts your plugins.
znap source zsh-users/zsh-history-substring-search 
# znap source marlonrichert/zsh-autocomplete
# znap source zsh-users/zsh-syntax-highlighting
znap source zdharma-continuum/fast-syntax-highlighting
znap source zsh-users/zsh-autosuggestions
znap source zsh-users/zsh-completions
znap source MichaelAquilina/zsh-you-should-use
