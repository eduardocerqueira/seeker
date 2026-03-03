#date: 2026-03-03T17:36:50Z
#url: https://api.github.com/gists/c116d0432e60d933e6ff54b777e495f9
#owner: https://api.github.com/users/vapvarun

#!/bin/bash
# Check PHP version via WP-CLI
wp --info | grep "PHP binary" -A1

# Or via PHP directly on server
php -v

# Check via WordPress admin header (if you have SSH access)
wp eval 'echo PHP_VERSION . "\n";'

# Verify active PHP via curl (phpinfo detection - needs a test file)
# NEVER leave phpinfo files on production. Use WP-CLI instead.
