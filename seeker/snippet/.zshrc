#date: 2022-01-04T17:18:46Z
#url: https://api.github.com/gists/cba42444b70ecc252a495560c2083cc1
#owner: https://api.github.com/users/dfelton

coffee() {
    ps aux | grep -v grep | grep caffeinate > /dev/null
    if [ $? -eq 1 ]; then
        caffeinate -dimsu &
    fi
}
killCoffee() {
    ps aux | grep -v grep | grep caffeinate > /dev/null
    if [ $? -eq 0 ]; then
        kill $(pgrep -f caffeinate)
    fi
}
