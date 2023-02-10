#date: 2023-02-10T17:10:46Z
#url: https://api.github.com/gists/1572d9a73bdaf2f5b6c20f44ba6da146
#owner: https://api.github.com/users/AugustUnderground

#!/bin/sh

ADR=https://www.coilcraft.com/api/power-inductor/parts

printf "Frequency,DCcurrent,RipplePercent,RippleCurrent,PeakCurrent,ACLoss\n"

for frq in $(seq 0.1 0.1 1); do
    for idc in $(seq 5 1 10); do
        for rip in $(seq 1 1 10); do
            ric="$(echo "$idc * $rip / 100" | bc -l | sed 's/^\./0./')"
            rpc="$(echo "$ric / 2 + $idc" | bc -l | sed 's/^\./0./')"
            jq ".current.idcCurrent = $idc" template.json | \
                jq ".current.rippleCurrent = $ric" | \
                jq ".current.rippleCurrentPercent = $rip" | \
                jq ".current.ipeakCurrent = $rpc" | \
                jq ".frequency.lower = $frq" | \
                jq ".frequency.value = $frq" > req.json
            
            acloss=$(curl -sH "Content-Type: application/json" --data @req.json $ADR \
                | gron | grep 'ACLoss' | tail -n 6 | head -n 1 | awk '{print $3}' | sed "s/;//g")

            printf "$frq,$idc,$rip,$ric,$rpc,$acloss\n"
        done
    done
done