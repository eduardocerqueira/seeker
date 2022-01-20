#date: 2022-01-20T17:03:05Z
#url: https://api.github.com/gists/cfd140549c7ffaa7b0c1c066247ae49d
#owner: https://api.github.com/users/guoquan

cat $1 | tr -d '\r' | sed 's/\\/\//g' |  md5sum -c -