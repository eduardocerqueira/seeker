#date: 2024-06-18T16:49:53Z
#url: https://api.github.com/gists/a2bdd5ed5481a1d5074e56d120265f79
#owner: https://api.github.com/users/ppcdias

# List All error_log Files
find /home/*/public_html -type f -name error_log -exec du -sh {} \;

# Clear All error_log Files
find /home/*/public_html -type f -iname error_log -delete

# Import a .sql File into a MySQL Database
mysql -u root -p dbname < /home/username/database.sql

# Get a List of the Most Frequent IP Addresses in Apache Logs
cat /usr/local/apache/logs/access_log | awk '{print $1}' | sort | uniq -c | sort -nr | head -n 20

# Execute a cPanel Backup Immediately
/usr/local/cpanel/bin/backup