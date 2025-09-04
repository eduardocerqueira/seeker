#date: 2025-09-04T17:09:24Z
#url: https://api.github.com/gists/9a461a57b1c81e5568d9f22ed019188c
#owner: https://api.github.com/users/rahmatsubandi

#!/bin/bash
# Script: cleanup_hestia.sh
# Purpose: Remove all HestiaCP files and configs for fresh install
# WARNING: This will delete all Hestia data, databases, emails, and configs.

echo "Stopping Hestia services..."
systemctl stop hestia 2>/dev/null
systemctl disable hestia 2>/dev/null
pkill -f hestia 2>/dev/null

echo "Removing Hestia directories..."
rm -rf /usr/local/hestia /etc/hestia /var/log/hestia /var/lib/hestia

echo "Removing Hestia database..."
mysql -e "DROP DATABASE IF EXISTS hestia;" 2>/dev/null

echo "Removing Hestia admin user and group..."
userdel -r admin 2>/dev/null
groupdel hestia 2>/dev/null

echo "Cleaning up mail, home directories, and Let's Encrypt..."
rm -rf /home/* /var/mail/* /etc/letsencrypt/*

echo "Removing Hestia cron jobs..."
crontab -r 2>/dev/null
rm -f /etc/cron.d/hestia* /etc/cron.daily/hestia*

echo "Cleanup complete. Server ready for fresh Hestia installation!"
echo "You can now run:"
echo "wget https://raw.githubusercontent.com/hestiacp/hestiacp/release/install/hst-install.sh"
echo "chmod +x hst-install.sh"
echo "bash hst-install.sh"
