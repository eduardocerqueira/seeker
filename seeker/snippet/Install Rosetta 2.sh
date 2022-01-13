#date: 2022-01-13T16:56:01Z
#url: https://api.github.com/gists/71fc1aa3c91f236c7967f6e9a336cf19
#owner: https://api.github.com/users/sgmills

#!/bin/sh

# Get architecture type
arch=$( /usr/bin/arch )
	
# If Apple silicon, install Rosetta 2
if [ "$arch" == "arm64" ]; then
	/usr/sbin/softwareupdate --install-rosetta --agree-to-license
fi
