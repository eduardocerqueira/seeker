#date: 2024-05-01T16:55:42Z
#url: https://api.github.com/gists/94f99bd6b4031f1545023ed6da216603
#owner: https://api.github.com/users/lucianmachado

find -maxdepth 5 -type d -name "projects" -prune -printf "%T@ %Tc %p\n" | sort -n