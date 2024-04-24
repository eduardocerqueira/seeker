#date: 2024-04-24T17:07:21Z
#url: https://api.github.com/gists/742860a3f30348f5b2dfd1c183fea9c3
#owner: https://api.github.com/users/palavarapuprakash44

#!/usr/bin/bash

echo "Listing browser's PIDs"
ps -e | grep firefox | awk '{print $1}'

echo "To display PPID of browser"
ps -e | grep firefox | awk '{print $1}' | head -1

echo "To stop the browser"
ps -e | grep firefox | awk '{print $1}' | head -1 |xargs  kill

echo "To list top 3 processes by CPU Usage"
top -o %CPU | head -10  # the first 7 lines are other matter

echo "To list top 3 processes by memory usage"
ps -o %MEM | head -10 # the first 7 lines are other matter

echo "To start python http sever on port 8000"
python3 -m http.server 8000

# to open another tab ctrl+shit+t
# to stop the previous process ctrl+c

echo "to start python HTTPS on port 90"
python3 -m http.server 90

echo "to display all active tcp and udp ports"
netstat -tulan

echo "to display process on port 5432"
ps -A | grep 5432