#date: 2025-02-17T16:44:57Z
#url: https://api.github.com/gists/3604ff5ba0692513cc068bc80a8747a2
#owner: https://api.github.com/users/seiyawut

#!/bin/bash

CONFIG_PATH="/etc/pgbackrest/pgbackrest.conf"

# Check for an existing pgbackrest config file and warn about overwriting
if [ -f "$CONFIG_PATH" ]; then
	echo "Existing pgbackrest configuration found at $CONFIG_PATH:"
	cat "$CONFIG_PATH"
	read -p "This will be overwritten. Continue? (y/n): " conf_answer
	if [ "$conf_answer" != "y" ]; then
		echo "Exiting..."
		exit 0
	fi
fi

# Prompt for required S3 environment variables if not set
for var in PGBR_S3_ENDPOINT PGBR_S3_KEY PGBR_S3_KEY_SECRET PGBR_S3_BUCKET; do
    if [ -z "${!var}" ]; then
        read -p "Enter value for $var: " value
        export "$var"="$value"
    fi
done

# Create new pgbackrest configuration file using the prompted values
cat <<EOF | sudo tee "$CONFIG_PATH" > /dev/null
[global]
repo1-type=s3
repo1-path=/pgbackrest
repo1-s3-uri-style=path
repo1-s3-endpoint=${PGBR_S3_ENDPOINT}
repo1-s3-region=auto
repo1-s3-key=${PGBR_S3_KEY}
repo1-s3-key-secret= "**********"
repo1-s3-bucket=${PGBR_S3_BUCKET}
repo1-cipher-type=none

[main]
pg1-path=/var/lib/postgresql/17/main
pg1-user=postgres
EOF
echo "New pgbackrest configuration created at $CONFIG_PATH"

if command -v pgbackrest >/dev/null 2>&1; then
	read -p "pgbackrest is already installed. Do you want to reinstall? (y/n): " answer
	if [ "$answer" != "y" ]; then
		echo "Exiting..."
		exit 0
	fi
fi

#make sure pg backrest is installed
pgbackrest --version

sudo apt update
sudo apt install -y postgresql-client libpq-dev libssl-dev libyaml-dev liblz4-dev libzstd-dev

sudo apt install -y curl
curl -1sLf https://pgbackrest.org/apt/pgbackrest-release-$(lsb_release -cs).deb -o pgbackrest-release.deb
sudo dpkg -i pgbackrest-release.deb
sudo apt update

sudo apt install -y pgbackrest

# Prompt to stop PostgreSQL and restore backup
read -p "Press enter to stop PostgreSQL and restore the backup..."
sudo systemctl stop postgresql
sudo -u postgres pgbackrest --stanza=main restore --type=standby

# Configure the instance as a standby server using demote-db.sh
# run script directly from the URL
curl -s https://gist.githubusercontent.com/seiyawut/3604ff5ba0692513cc068bc80a8747a2/raw/19fdd65fad4c0ef0902bdbd7a58d27dd9525cfec/demote-db.sh | bash

# Start PostgreSQL
sudo systemctl start postgresql
echo "PostgreSQL is now running as a standby server."
erver."
