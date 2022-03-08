#date: 2022-03-08T16:51:24Z
#url: https://api.github.com/gists/9b5eb54850fc4a15813a1d4f87ff85fb
#owner: https://api.github.com/users/okuramasafumi

if [[ $- =~ i ]]; then
  bind '"\er": redraw-current-line'
  bind '"\C-g\C-f": "$(_gf)\e\C-e\er"'
  bind '"\C-g\C-b": "$(_gb)\e\C-e\er"'
  bind '"\C-g\C-t": "$(_gt)\e\C-e\er"'
  bind '"\C-g\C-h": "$(_gh)\e\C-e\er"'
  bind '"\C-g\C-r": "$(_gr)\e\C-e\er"'
  bind '"\C-g\C-s": "$(_gs)\e\C-e\er"'
fi