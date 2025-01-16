#date: 2025-01-16T17:00:59Z
#url: https://api.github.com/gists/750fdb86def902b4b0adf5e2009203b4
#owner: https://api.github.com/users/tranphuquy19

#!/bin/bash

#
# RAM
#

echo "RAM"
echo "---"

SUM=0
SUM_ACTUAL=0

while read vm
do
    if [ ! -z "$vm" ]; then
        MAX_MEM=$(($(virsh dominfo $vm | grep "Max memory" | cut -f 7 -d " ") / 1024))
        
        ACTUAL_MEM=$(virsh dommemstat $vm 2>/dev/null | grep rss | awk '{print $2/1024}')
        
        if [ ! -z "$ACTUAL_MEM" ]; then
            printf "%-25s = %'.0f MiB (Actual: %'.0f MiB)\n" $vm $MAX_MEM $ACTUAL_MEM
            SUM=$((SUM + MAX_MEM))
            SUM_ACTUAL=$(awk "BEGIN {print $SUM_ACTUAL + $ACTUAL_MEM}")
        else
            printf "%-25s = %'.0f MiB (Offline)\n" $vm $MAX_MEM
            SUM=$((SUM + MAX_MEM))
        fi
    fi
done < <(virsh list --all --name)

printf "\nTotal Allocated: %'.0f MiB\n" $SUM
printf "Total Actual Used: %'.0f MiB\n\n" $SUM_ACTUAL

#
# CPUs
#

echo "CPU(s)"
echo "------"

SUM=0

while read vm
do
    if [ ! -z "$vm" ]; then
        USED=$(virsh dominfo $vm | grep "CPU(s)" | cut -f 10 -d " ")
        printf "%-25s = %'.0f cpu(s)\n" $vm $USED
        SUM=$((SUM + USED))
    fi
done < <(virsh list --all --name)

printf "\nTotal: %'.0f CPU(s)\n" $SUM