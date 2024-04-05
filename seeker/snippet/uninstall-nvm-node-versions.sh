#date: 2024-04-05T16:47:43Z
#url: https://api.github.com/gists/43a0042f5e269d4347b2656064e38a32
#owner: https://api.github.com/users/jamescallumyoung

# Note: This script will not uninstall the version currently in use.
#       All other versions will be uninstalled.

nvm ls --no-colors --no-alias | xargs | tr " " "\n" | grep ^v | while read line ; do nvm uninstall $line ; done
