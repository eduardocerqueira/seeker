#date: 2024-04-12T16:44:52Z
#url: https://api.github.com/gists/2561d0685928d70eb20b634737f1e99e
#owner: https://api.github.com/users/rvmaher

#!/bin/bash

# Function to display cleanup messages with timestamp
print() {
  timestamp=$(date +%r)
  echo "***** $1 @ $timestamp *****"
}

# Clean npm cache
print "Cleaning npm cache..."
npm cache clean --force

# Clean Node.js related files and directories
print "Cleaning Node.js related files and directories..."
rm -rf node_modules
rm -rf package-lock.json

# Clean Watchman
print "Resetting Watchman..."
watchman watch-del-all

# Reinstall npm packages
print "Reinstalling npm packages..."
npm install

print "JavaScript-related files, Metro bundler, Watchman, and npm packages reinstalled successfully!"
