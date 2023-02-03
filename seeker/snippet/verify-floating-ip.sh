#date: 2023-02-03T16:47:05Z
#url: https://api.github.com/gists/e33e4146977d8ed69b9cf8a5d3f20bac
#owner: https://api.github.com/users/rjchicago

#!/bin/sh
set -Eeuo pipefail
echo

: ${NAMESPACE:?NAMESPACE is required}
: ${SERVICE:?SERVICE is required}
: ${FLOATING_IP:?FLOATING_IP is required}

echo "VERIFY FLOATING IP: $FLOATING_IP"
while true; do
    ASSIGNED_IP=$(kubectl get svc -n $NAMESPACE $SERVICE -o=jsonpath='{.status.loadBalancer.ingress[0].ip}')
    [[ "$ASSIGNED_IP" = "$FLOATING_IP" ]] && exit 0
    echo " ..."
    sleep 1
done

### EXAMPLE USAGE ###
# export FLOATING_IP="1.2.3.4"
# export NAMESPACE="my-namespace"
# export SERVICE="my-service"
# verification_passed() { echo " ✅ VERIFICATION PASSED"; }
# verification_failed() { echo " 💀 VERIFICATION FAILED" && exit 1; }
# timeout 5s $(dirname $0)/verify-floating-ip.sh && verification_passed || verification_failed



