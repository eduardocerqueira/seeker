#date: 2021-09-10T16:51:34Z
#url: https://api.github.com/gists/2cbbd0801770dff2e104ff1ed3f68316
#owner: https://api.github.com/users/recall704

ip rule add fwmark 0x233 lookup 100
ip route add local 0.0.0.0/0 dev lo table 100