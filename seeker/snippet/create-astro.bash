#date: 2025-10-17T16:55:16Z
#url: https://api.github.com/gists/f77dcbded5924d7631b344071a85f454
#owner: https://api.github.com/users/Arxcis

# create-astro.bash
set -e

#
# Usage example:
#
# bash create-astro.bash PROJECT_NAME
#
PROJECT_NAME=${1:-"YOU-FORGOT-TO-GIVE-YOUR-PROJECT-A-NAME"}


# 1. Creating Astro project...
npm create astro@latest "$PROJECT_NAME"
cd "$PROJECT_NAME"


# 2. Creating .htaccess file...
cat <<HTACCESS > ./public/.htaccess

ErrorDocument 404 /404.html
ErrorDocument 500 /500.html

HTACCESS


# 3. Creating 404.astro page...
cat <<404 > ./src/pages/404.astro

<p>404 - Page Not Found</p>

404


# 4. Creating 500.astro page...
cat <<500 > ./src/pages/500.astro

<p>500 - Internal Server Error</p>

500


# osv....

