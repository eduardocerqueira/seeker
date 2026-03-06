#date: 2026-03-06T17:11:17Z
#url: https://api.github.com/gists/2cfbc1c5b4ae176b89fd3f70cbf14bc4
#owner: https://api.github.com/users/mercan798

  GNU nano 7.2                                                                                                       reload.sh                                                                                                                
#!/bin/bash

while true; do
    clear
    echo
    echo "  __  __                              __        __           __    __"
    echo " /  |/  |  ___   ____  ____ ___ ___ / /  _    / /_  _____  / /___/ /"
    echo "/ /|_/ /  / -_) / __/ / __// _ \`// // _ \   / /\ \/ / _ \/ // _  /"
    echo "/_/  /_/   \__/ /_/   \__/ \_,_//_//_.__/  /_/ /_/\_\\___/_/ \_,_/"
    echo

    echo "Uptime: $(uptime -p)"
    echo "CPU Kullanımı: $(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')%"
    echo "RAM Kullanımı: $(free -h | awk '/Mem:/ {print $3 "/" $2}')"
    echo "Disk Kullanımı: $(df -h / | awk 'NR==2 {print $3 "/" $2}')"

    echo
    echo "Yenilemek için Enter'a bas, çıkmak için Ctrl+C"
    read
done


