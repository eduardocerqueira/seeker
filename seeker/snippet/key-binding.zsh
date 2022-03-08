#date: 2022-03-08T16:51:24Z
#url: https://api.github.com/gists/9b5eb54850fc4a15813a1d4f87ff85fb
#owner: https://api.github.com/users/okuramasafumi

join-lines() {
  local item
  while read item; do
    echo -n "${(q)item} "
  done
}

() {
  local c
  for c in $@; do
    eval "fzf-g$c-widget() { local result=\$(_g$c | join-lines); zle reset-prompt; LBUFFER+=\$result }"
    eval "zle -N fzf-g$c-widget"
    eval "bindkey '^g^$c' fzf-g$c-widget"
  done
} f b t r h s c