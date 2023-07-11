#date: 2023-07-11T16:47:12Z
#url: https://api.github.com/gists/392bbf003bdd25ec3b7a24223c8627ad
#owner: https://api.github.com/users/lorenzom222

#!/bin/bash

# Run ./flash4 10 times
# Run ./flash4 10 times
for i in {1..10}
do
    ./flash4
done

# Get each line that has "seconds in monitoring period"
nums=$(grep "seconds in monitoring period" sedov.log |  awk '{print $NF}')

total=0
count=0
for n in $nums
do

    time=$(echo $n)
    echo "Time: $time"
    total=$(expr $(printf "%.0f" $total) + $(printf "%.0f" $time))
    echo "Sum = $total " 
    count=$((count+1))
done

average=$(expr $total / $count)

echo "The average time is $average seconds."

