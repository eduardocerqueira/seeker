#date: 2024-04-24T17:06:08Z
#url: https://api.github.com/gists/8b78b0695d019e3164b56c5650f07af0
#owner: https://api.github.com/users/561sharath

#!/usr/bin/bash

echo "list browser process ids and parent ids"

ps -e | grep firefox | awk '{print $1}'

echo "stop the browser application"

ps -e | grep firefox | awk '{print $1}' | head -1

echo "kill the browser"

ps -e | grep firefox | awk '{print $1}' | head -1 | xargs kill

echo "top 3 process by cpu usage"

ps -eo pid,comm,%cpu --sort=-%cpu | head -n 4

top -o %CPU | head -n 3

echo "top 3 process by memory usage"

ps -eo pid,comm,%cpu --sort=-%mem | head -n 4

ps -o %MEM | head -n 4

echo "start a python HTTP server on port 8000"

python3 -m http.server 8000

echo "stop the server in another tab"

pgrep -f 'python3 -m http.server' | head -n 1 | xargs kill

echo "start python HTTP server on port 90"

sudo python3 -m htttp.server 90

echo "display all activtive connection of TCP/UDP ports"

ss -tuln

netstat -tulan

echo "install htop,vim,nginx"

sudo apt update

sudo apt install htop

sudo apt install vim

sudo apt install nginx

echo "uninstall nginx"

sudo apt remove --purge nginx ngnix-common

echo "local ip address"

ip addr show

echo "ip address of google.com"

nslookup google.com

ping google.com | head -n 4

echo "internet is working in cli"

curl google.com





