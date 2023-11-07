#date: 2023-11-07T17:09:55Z
#url: https://api.github.com/gists/0ea32c6eb2f4bc16c76962f54d437ae3
#owner: https://api.github.com/users/gonoph

#!/bin/bash
# vim: sw=4 ts=4 expandtab
# modifed from https://www.baeldung.com/linux/read-process-memory

if [ -z "$1" ]; then
    echo "Usage: $0 <pid>"
    exit 1
fi
if [ ! -d "/proc/$1" ]; then
    echo "PID $1 does not exist"
    exit 1
fi

while read -r mem_range perms JUNK ; do
    if [[ "$perms" == "r"* ]]; then
        IFS="-" read start_addr end_addr JUNK <<< "$mem_range"
        start_addr="$(( 16#$start_addr ))"
        end_addr="$(( 16#$end_addr ))"

        echo "Reading memory range $mem_range: $start_addr to $end_addr"
        length=$(( $end_addr - $start_addr))
        # dd if="/proc/$1/mem" of="/dev/stdout" bs=1 skip="$start_addr" count="$((16#$end_addr - 16#$start_addr))" 2>/dev/null
        echo "hexdump -C -n $length -s $start_addr < /proc/$1/mem"
        hexdump -C -n $length -s $start_addr < /proc/$1/mem
    fi
done < "/proc/$1/maps"