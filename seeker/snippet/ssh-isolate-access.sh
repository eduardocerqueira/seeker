#date: 2022-03-30T16:55:00Z
#url: https://api.github.com/gists/cd0354c5b204001166c8446c410cb977
#owner: https://api.github.com/users/dougpagani

ssh-isolate-access() {
    cat ~/.ssh/id_rsa.pub | command ssh "$@" '
    cat > ~/newakeys;
    nohup bash -c '"'"'trap "" HUP; echo $$ > ~/killme && sleep 10 && mv -f ~/authorized_keys ~/.ssh/authorized_keys;'"'"' </dev/null &>/dev/null & disown;
    mv ~/.ssh/authorized_keys ~/authorized_keys &&
    mv ~/newakeys ~/.ssh/authorized_keys;
    echo DONE: close connection
'
}