#date: 2025-04-23T17:08:14Z
#url: https://api.github.com/gists/66abdad211c204189a7f6f9b379f6560
#owner: https://api.github.com/users/uturuncoglu

#!/bin/bash

# executable name (modifiable by user)
exename="fv3.exe"

# get list of nodes from jobid
lst_nodes=`qstat -n -1 $1 | awk '{print $12}' | grep '/' | tr '+' '\n' | awk -F\/ '{print $1}'`

# loop over nodes and collect trace from each process
for i in $lst_nodes
do
  ssh -n -f $i "sh -c 'cd `pwd`; nohup ./trace_cmd.sh `pwd` $exename $i > /dev/null 2>&1 &'"
done