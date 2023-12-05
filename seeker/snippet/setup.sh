#date: 2023-12-05T17:10:12Z
#url: https://api.github.com/gists/f3b03b93f6397c43c5a85a9d909af035
#owner: https://api.github.com/users/NoirPi

#!/bin/bash

# Check if the auth database file exists
if [ ! -e /var/lib/ntfy/user.db ]; then
    # Database file doesn't exist, perform setup
    ntfy user add --role=admin admin_user
    ntfy user add --role=user read_write_user
    ntfy user add --role=user read_only_user
    ntfy user add --role=user write_only_user

    echo "Setup completed!"
fi

# Start the ntfy server
ntfy serve --log-level warn --listen-http :8080
