#date: 2021-10-28T17:01:42Z
#url: https://api.github.com/gists/4a20ba10c26ac2ad02cb0425b8b0f826
#owner: https://api.github.com/users/nickdesaulniers

#!/usr/bin/env sh
# Usage: ./measure_30.sh 'code to measure' 'setup code not to measure'
eval $2
tempfile=$(mktemp)
count=30
for i in $(seq 1 $count); do
  perf stat -B -e cycles:u $1 2>&1 1>/dev/null | grep cycles:u | tr -s ' ' | cut -d ' ' -f 2 | tr -d , | tee --append $tempfile
done
total=0
for i in $(awk '{ print $1; }' $tempfile); do
  total=$(echo $total+$i | bc)
done
echo "scale=2; $total / $count" | bc | xargs printf "Average of 30 runs: %s cycles\n"