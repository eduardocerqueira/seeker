#date: 2024-04-05T16:47:43Z
#url: https://api.github.com/gists/43a0042f5e269d4347b2656064e38a32
#owner: https://api.github.com/users/jamescallumyoung

# This script removes the Yarn Classic "global folder" which houses globally installed packages, installed by Yarn Classic.
# Note: Yarn Modern does not use this folder.

# TEST THIS

rm -rf $(yarn global bin)/..