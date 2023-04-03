#date: 2023-04-03T17:05:25Z
#url: https://api.github.com/gists/452d23147e4722df245db18bc8e66cc6
#owner: https://api.github.com/users/TheFutonEng

# Disconnect from the internet, script must be run with 
# root privileges.
ip link add airgap type dummy
ip link set airgap up
ip -c address add 1.1.1.1/16 dev airgap
ip route add default via 1.1.1.1 dev airgap metric 1