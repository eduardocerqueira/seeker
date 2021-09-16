#date: 2021-09-16T16:56:40Z
#url: https://api.github.com/gists/07b1c9617c2ed6b987031995082db230
#owner: https://api.github.com/users/amieiro

#!/usr/bin/env bash

# Constants. Update it as you need
GLOTPRESS_BASE_DIR=/Users/amieiro/code/wordpress/wp/wp-content/plugins/GlotPress-WP
WP_TEST_INSTALL_BASE_DIR=/Users/amieiro/code/wordpress/tests
DB_NAME=wp_test
DB_USER=wp_test
DB_PASSWORD=password
DB_HOST=localhost
WP_VERSION=latest
SKIP_DATABASE_CREATION=false

# Copy and paste the next 3 lines in your shell, so you have them in the current shell
export TMPDIR=$WP_TEST_INSTALL_BASE_DIR/tmpdir/
export WP_TESTS_DIR=$WP_TEST_INSTALL_BASE_DIR/wptestdir/
export WP_CORE_DIR=$WP_TEST_INSTALL_BASE_DIR/wpcoredir/
# printenv

sudo rm -rf $TMPDIR
sudo rm -rf $WP_TESTS_DIR
sudo rm -rf $WP_CORE_DIR

mkdir -p $TMPDIR

# install-wp-tests.sh <db-name> <db-user> <db-pass> [db-host] [wp-version] [skip-database-creation]
$GLOTPRESS_BASE_DIR/bin/install-wp-tests.sh $DB_NAME $DB_USER $DB_PASSWORD $DB_HOST $WP_VERSION $SKIP_DATABASE_CREATION
