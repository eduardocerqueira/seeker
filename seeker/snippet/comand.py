#date: 2024-05-28T17:01:50Z
#url: https://api.github.com/gists/6fc0e3f26c901568eba79f3a97485ba9
#owner: https://api.github.com/users/liviuxyz-ctrl

import os

# Retrieve the current user's name from environment variables
current_user = os.environ.get('USERNAME')

# Print the current user's name
print("Current logged-in user:", current_user)
