#date: 2023-11-09T16:47:48Z
#url: https://api.github.com/gists/fff1a16e34d40bfc4c7c390d16096b57
#owner: https://api.github.com/users/tnn4

#!/usr/bin/env python3

# Usage: python3 sysmon.py /
# # Pass in the disk you want to check, here we pass in root(/)

# Setup
# You need to install psutil
# pip3 install psutil

# Error Missing Module

# if you just installed psutil and python says it's missing you'll need to look for the path to your libs:
# Check which pip3 you have:
# pip3 --version
# it should give you a path to your site packages
# export that with PYTHONPATH
# E.g. export PYTHONPATH=/home/t/.local/lib/python3.10/site-packages:${PYTHONPATH}

# # upgraded to python3 from this great project: https://github.com/ckinateder/sysmon-1.0.1

import psutil, os, sys

import time

mem_perc = 0 #init var
swap_perc = 0 #init var
mbytes_sent = 0 #init var
mbytes_recv = 0 #init var
cpu_perc = 0 #init var
swap = 0 #init var
mem = 0 #init var
net = 0 #init var


def disp(degree):
    global cpu_perc
    global swap
    global swap_perc
    global mem
    global mem_perc
    global net
    global mbytes_sent
    global mbytes_recv

    cpu_perc = psutil.cpu_percent(interval=1, percpu=True)
    swap = psutil.swap_memory()
    swap_perc = swap.percent
    mem = psutil.virtual_memory()
    mem_perc = mem.percent
    net = psutil.net_io_counters()
    mbytes_sent = float(net.bytes_sent) / 1048576
    mbytes_recv = float(net.bytes_recv) / 1048576

    os.system('clear') #clear the screen

    print("System Monitor")
    
    ##
    ## CPU
    ##
    print("-"*30)
    print("CPU")
    print("-"*30)
    print("CPU Temperature: " , degree, "'C")
    for i in range(len(cpu_perc)):
        print ("CPU Core", str(i+1),":", str(cpu_perc[i]), "%")

    ##
    ## RAM
    ##
    print("-"*30)
    print("MEMORY")
    print("-"*30)
    
    print(f"RAM: {mem_perc}")
    print(f"Swap: {swap_perc}")
    print("-"*30)
    
    ##
    ## NETWORK
    ##
    print("NETWORK")  
    print("-"*30 )   
    print(f"MB sent: {mbytes_sent}")   
    print(f"MB received: {mbytes_recv}")   
    print("-"*30)   
    
    print("DISKS")   
    print("-"*30)    

    if len(sys.argv) > 1:
        for disk in range(1, len(sys.argv)):
            tmp = psutil.disk_usage(sys.argv[disk])
            print(sys.argv[disk], "\n")
            print("Megabytes total: ")
            print(str(float(tmp.total) / 1048576))
            print("Megabytes used: ")
            print(str(float(tmp.used) / 1048576))
            print("Megabytes free: ")
            print(str(float(tmp.free) / 1048576))
            print("Percentage used: ")
            print(tmp.percent, "\n")

def main():
    print("Press Ctrl+C to exit")
    while True:
        temp = open("/sys/class/thermal/thermal_zone0/temp").read().strip().lstrip('temperature :').rstrip(' C')
        temp = float(temp) / 1000
        disp(temp)
        time.sleep(0.5)

main()