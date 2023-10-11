#date: 2023-10-11T17:03:37Z
#url: https://api.github.com/gists/92e1fe46656a3b66e302ed3fede6aac8
#owner: https://api.github.com/users/leshikus

# specific settings go here

export HTTPS_PROXY="$https_proxy"

test -L sh.sh || ln -s sc.sh sh.sh

cd

cat <<EOF >.vimrc
syntax on
set tabstop=4 expandtab
EOF

if test $(basename "$0") = sc.sh
then
    screen -r -d -S main || screen -S main bash
else
    bash
fi