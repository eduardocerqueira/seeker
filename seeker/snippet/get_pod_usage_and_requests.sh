#date: 2025-07-21T16:50:36Z
#url: https://api.github.com/gists/cddffd35351c30d727b43691574c914f
#owner: https://api.github.com/users/juanmancebo

#!/bin/bash

{
  echo -e "NAMESPACE\tPOD\tCPU_REQ(m)\tCPU_USAGE(m)\tMEM_REQ(Mi)\tMEM_USAGE(Mi)"
  for ns in <NAMESPACE1> <NAMESPACE2>; do           
    kubectl top pods -n "$ns" --no-headers | while read pod cpu_usage mem_usage; do
      cpu_req=$(kubectl get pod "$pod" -n "$ns" -o jsonpath='{.spec.containers[*].resources.requests.cpu}' | \
        awk '{sum=0; for(i=1;i<=NF;i++) { val=$i; if(val ~ /m$/) {val=val+0} else {val=val*1000} sum+=val;} print sum}')
      mem_req=$(kubectl get pod "$pod" -n "$ns" -o jsonpath='{.spec.containers[*].resources.requests.memory}' | \
        awk '{sum=0; for(i=1;i<=NF;i++) { val=$i; if(val ~ /Mi$/) {val=val+0} else if(val ~ /Ki$/) {val=val/1024} else if(val ~ /Gi$/) {val=val*1024} sum+=val;} print sum}')

      cpu_usage_num=$(echo $cpu_usage | sed 's/m//')
      mem_usage_num=$(echo $mem_usage | sed 's/Mi//')

      echo -e "$ns\t$pod\t$cpu_req\t$cpu_usage_num\t$mem_req\t$mem_usage_num"
    done
  done
} | column -t -s $'\t'