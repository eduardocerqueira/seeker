#date: 2024-02-02T16:49:58Z
#url: https://api.github.com/gists/a83eaa00cd745a046f5dbe4e8bb8a6d9
#owner: https://api.github.com/users/jameswilson

#!/bin/bash
# Deployment script for Acquia Cloud Next on a multisite (non-Site Factory) setup.
# Place this file in your repository at hooks/common/post-code-deploy/deploy.sh
# See https://github.com/acquia/cloud-hooks/blob/master/samples/post-code-deploy.tmpl

site="$1"
target_env="$2"

cd "/var/www/html/$site.$target_env/docroot"

# Take all sites offline.
for dir in sites/*/; do
  multisite_name=$(basename "$dir")
  DRUSH="../vendor/bin/drush --uri=$multisite_name"
  set -x # Expose the following command output to Acquia Cloud's Task Log.
  $DRUSH state:set system.maintenance_mode 1 --input-format=integer
  { set +x; } 2>&- # Hide the rest of the script output.
done

# Update each site and bring it back online as early as possible.
# @todo parallelize update tasks across multisites using `parallel` command.
for dir in sites/*/; do
  multisite_name=$(basename "$dir")
  DRUSH="../vendor/bin/drush --uri=$multisite_name"
  set -x # Expose the following command output to Acquia Cloud's Task Log.
  $DRUSH cache:rebuild
  $DRUSH updatedb
  $DRUSH config:import --yes
  $DRUSH config:import --yes
  $DRUSH cohesion:import
  $DRUSH sitestudio:package:import --yes
  $DRUSH state:set system.maintenance_mode 0 --input-format=integer
  $DRUSH cache:rebuild

  { set +x; } 2>&- # Hide the rest of the script output.
done