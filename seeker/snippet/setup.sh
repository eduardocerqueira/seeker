#date: 2025-03-31T16:54:29Z
#url: https://api.github.com/gists/a07fcf730889dfe564f13845256da804
#owner: https://api.github.com/users/qnnnnez

#!/bin/bash

# tc direct-action mode: https://qmonnet.github.io/whirl-offload/2020/04/11/tc-bpf-direct-action/

tc qdisc add dev eth0 clsact
tc filter add dev eth0 ingress bpf direct-action obj pktsize_hist_bpfel.o
tc filter add dev eth0 egress bpf direct-action obj pktsize_hist_bpfel.o