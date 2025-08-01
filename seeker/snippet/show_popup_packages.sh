#date: 2025-08-01T17:14:34Z
#url: https://api.github.com/gists/d70d715b359dd1b2e8f280046253dc93
#owner: https://api.github.com/users/lucasbracher

#!/bin/bash
#
# This script is really useful when you have malware in your phone and you want to remove it.
#
# Before you begin, you need to install adb tools (look for instructions for your platform) and
# you need to enable developer mode in your phone.
#
# After it, connect your phone to your computer, enable ADB debug, enable the connection between
# phone and computer, and then you are able to start.
#
# 1. dump all packages in a file:
adb shell pm list packages | sed 's/package://' > packages.txt
# 2. wait for the moment the popup shows and run this:
adb shell dumpsys window windows > dump.txt
for each in $(cat packages.txt); do grep --color=auto $each dump.txt; done
# 3. inspect the colored output and use your good sense to spot the malware
# 4. run this command:
adb shell pm uninstall -k --user 0 <package_name>