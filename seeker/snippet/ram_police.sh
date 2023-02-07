#date: 2023-02-07T17:05:22Z
#url: https://api.github.com/gists/47100a896d0d265aeb589b292919d744
#owner: https://api.github.com/users/MiniXC

free_ram=$(free -g | grep -m1 -E "[[:digit:]]" | tail -c4)
ram_limit=10

# check if availabe RAM is below the limit every second
# do nothing if RAM is above the limit
# kill all python processes if RAM is below the limit
while true; do
    if [ $free_ram -lt $ram_limit ]; then
        echo "RAM is below the limit"
        echo "Killing all python processes"
        pkill -f .*python.*
    else
        echo -ne "RAM $free_ram is above the limit\033[0K\r"
    fi
    sleep 1
done