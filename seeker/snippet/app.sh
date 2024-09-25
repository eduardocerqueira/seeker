#date: 2024-09-25T16:58:54Z
#url: https://api.github.com/gists/25329215daef9137e78c32771e924b10
#owner: https://api.github.com/users/ernestoruiz89

#!/bin/bash

# Check if an argument (application name) is provided
if [ -z "$1" ]; then
    echo "You must provide the application name. Usage: ./app.sh <app_name>"
    exit 1
fi

# The application to search for is passed as an argument
app_name=$1

# Navigate to your Bench installation directory
cd /bench/frappe/frappe-bench

# Get a list of all the sites
sites=$(ls sites/)

# Variable to track if the app is found in any site
app_found=false

# Iterate over each site and check if the application is installed
for site in $sites; do
    if bench --site $site list-apps | grep -q $app_name; then
        echo "The application '$app_name' is installed on site '$site'."
        app_found=true
    fi
done

# If the app was not found on any site
if [ "$app_found" = false ]; then
    echo "The application '$app_name' is not installed on any site."
fi
