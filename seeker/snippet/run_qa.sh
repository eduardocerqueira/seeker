#date: 2022-06-07T17:06:30Z
#url: https://api.github.com/gists/2c20a9e0d4e7ca7da2497d818bb6e368
#owner: https://api.github.com/users/majek

#!/bin/bash

BPID="1111111111"

sysctl net.core.rmem_max=$[32*1024*1024]

## should be default
### sysctl net.core.wmem_max=$[212992]
ethtool -K lo tso off

function finish {
    echo "[-] Cleaning up"
    kill $BPID
    echo
}
trap finish EXIT

~marek/bin/bpftrace -q tcp_qa.bt -o bt-slow.txt &
BPID="$!"
sleep 1
python3 window_qa.py slow
kill $BPID


~marek/bin/bpftrace -q tcp_qa.bt -o bt-fast.txt &
BPID="$!"
sleep 1
python3 window_qa.py fast
kill $BPID

sed '/Lost/d' bt-slow.txt|egrep -v "^@"|sort -n|sponge bt-slow.txt
sed '/Lost/d' bt-fast.txt|egrep -v "^@"|sort -n|sponge bt-fast.txt


/usr/bin/gnuplot -c ./plot_qa.gnuplot \
    bt-slow.txt \
    bt-fast.txt \
    out.png
