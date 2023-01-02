#date: 2023-01-02T16:58:44Z
#url: https://api.github.com/gists/bac12a68e0e4d57f3a844fa20c5dee0c
#owner: https://api.github.com/users/MikeJansen

# Pipe YAML or JSON, optionally query it, and display as colorized YAML in less
function y() { yq -C "$@" | less -R; }
function yj() { yq -C -p=j -o=y "$@" | less -R; }
