#date: 2025-08-28T17:00:36Z
#url: https://api.github.com/gists/b97ad2391b7f3613c8e36e29c6aa5f96
#owner: https://api.github.com/users/kagg-design

#!/bin/bash
#set -v

WP_PATH=$(wp config path)

if [[ ! $WP_PATH ]]; then
  echo "This is not a WordPress install"
  exit 1
fi

WP_PATH="$(dirname "$WP_PATH")"

PARAMETERS_FILE=""$WP_PATH/sync.params""

if [[ ! -f "$PARAMETERS_FILE" ]]; then
  echo "Parameters file $PARAMETERS_FILE does not exist"
  exit 1
fi

TYPE="$1"
if [[ -z $TYPE ]]; then
  TYPE="db"
fi

if [[ "db" != "$TYPE" && "plugins" != "$TYPE" && "uploads" != "$TYPE" && "all" != "$TYPE" ]]; then
  echo "Argument can be 'db', 'plugins', 'uploads' or 'all' only"
  exit 1
fi

# shellcheck disable=SC1090
if ! source "$PARAMETERS_FILE";
then
  exit 1
fi

if [[ "" == "$REMOTE_USER" ]]; then
  REMOTE_USER=$USER
fi

SECONDS=0

if [[ "db" == "$TYPE" || "all" == "$TYPE" ]]; then
  echo -e "\nDownloading database..."
  ssh -l "$REMOTE_USER" -i "$KEY_FILE" "$REMOTE_DOMAIN" "cd $REMOTE_PATH > /dev/null&&/usr/local/bin/wp db export /tmp/wp.sql > /dev/null"
  scp -i "$KEY_FILE" "$REMOTE_USER"@"$REMOTE_DOMAIN":/tmp/wp.sql wp.sql
  ssh -l "$REMOTE_USER" -i "$KEY_FILE" "$REMOTE_DOMAIN" "rm /tmp/wp.sql"
  echo -e "Done."

  echo -e "\nImporting database..."
  wp --quiet db import wp.sql
  echo -e "Done."

  rm wp.sql

  echo -e "\nReplacing domain in database..."
  wp search-replace --url="$REMOTE_URL" "$REMOTE_URL" "$LOCAL_URL" --recurse-objects --report-changed-only --precise --skip-columns=guid --skip-tables=wp_users --skip-plugins --skip-themes --allow-root
  wp cache flush
  echo -e "Done database sync."
fi

echo -e "\nDoing after db sync commands..."
eval $AFTER_SYNC_COMMANDS
echo -e "Done after db sync commands."

if [[ "plugins" == "$TYPE" || "all" == "$TYPE" ]]; then
  echo -e "\nSyncing plugins..."
  IFS=' '
  read -ra PLUGINS <<< "$SYNC_PLUGINS"
  for PLUGIN in "${PLUGINS[@]}"; do
    rsync -ase "ssh -l $REMOTE_USER -i $KEY_FILE" "$REMOTE_USER"@"$REMOTE_DOMAIN":"$REMOTE_PATH"/wp-content/plugins/"$PLUGIN"/* "$LOCAL_PATH"/wp-content/plugins/"$PLUGIN"
  done
  echo -e "Done."
fi

if [[ "uploads" == "$TYPE" || "all" == "$TYPE" ]]; then
  echo -e "\nSyncing uploads..."
  rsync -ase "ssh -l $REMOTE_USER -i $KEY_FILE" --exclude "*backwpup*/*" "$REMOTE_USER"@"$REMOTE_DOMAIN":"$REMOTE_PATH"/wp-content/uploads/* "$LOCAL_PATH"/wp-content/uploads
  echo -e "Done."
fi

echo -e "\nFinished. Time elapsed: $SECONDS seconds."
