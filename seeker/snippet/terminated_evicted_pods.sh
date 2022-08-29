#date: 2022-08-29T17:03:05Z
#url: https://api.github.com/gists/c4792d4f8a5518874d8f34f594fbc8b6
#owner: https://api.github.com/users/chalisekrishna418

NAMESPACES=$(kubectl get pods -A | grep Evicted | cut -f1 -d " " | uniq)
for NS in $NAMESPACES
do
	PODS=$(kubectl get pods -n $NS | grep Evicted | cut -f1 -d " ")
	i=0
	for POD in "$PODS"
	do
	  kubectl delete pods -n $NS $POD
	  ((i=i+1))
	  echo "+++++++Deleted $i pods +++++++"
	done
done
