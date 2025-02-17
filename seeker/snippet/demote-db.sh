#date: 2025-02-17T16:44:57Z
#url: https://api.github.com/gists/3604ff5ba0692513cc068bc80a8747a2
#owner: https://api.github.com/users/seiyawut

#!/bin/bash

# Define PostgreSQL config directory and file
PG_CONF="/etc/postgresql/17/main/postgresql.conf"
PG_DATA="/var/lib/postgresql/17/main"

echo "⚠️  Stopping PostgreSQL to demote..."
systemctl stop postgresql@17-main

echo "✅ PostgreSQL stopped."

# Backup the current config file
cp "$PG_CONF" "$PG_CONF.bak"

echo "🔧 Updating PostgreSQL configuration for standby mode..."

echo "🔄 Performing pgbackrest standby restore..."
if ! sudo pgbackrest --stanza=main --type=standby --delta restore; then
  echo "❌ Restore failed!"
  exit 1
fi
echo "✅ Restore completed successfully."

# Modify WAL settings for standby mode
sed -i "s/^wal_level.*/wal_level = replica/" "$PG_CONF"
sed -i "s/^archive_mode.*/archive_mode = off/" "$PG_CONF"

# Update archive_command for standby mode
sed -i "/^archive_command/d" "$PG_CONF"
echo "archive_command = 'pgbackrest --stanza=main archive-get %f %p'" >> "$PG_CONF"

# Ensure standby mode is enabled
rm -f "$PG_DATA/recovery.signal"
touch "$PG_DATA/standby.signal"

echo "✅ PostgreSQL is now configured for standby mode."

# Start PostgreSQL in standby mode
echo "🚀 Starting PostgreSQL in standby mode..."
systemctl start postgresql@17-main

echo "🎉 PostgreSQL is now running as a standby server!"
