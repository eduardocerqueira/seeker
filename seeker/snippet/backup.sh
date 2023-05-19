#date: 2023-05-19T16:46:44Z
#url: https://api.github.com/gists/0a61db668da7ca3539782c12c7efca1f
#owner: https://api.github.com/users/jorgeteixe

#!/bin/sh
# autor: Álvaro Freire Ares
# autor: Jorge Teixeira Crespo

BACKUP_DIR="/var/backups"

while IFS=':' read -r username _ _ _ _ homedir _; do
  if [ -n "$homedir" ] && [ -f "$homedir/.IWantBackup" ]; then
    backup_file="${BACKUP_DIR}/$(date +%Y%m%d)_${username}.tar.gz"
    if [ ! -f "$backup_file" ]; then
      tar -czf "$backup_file" -C "$(dirname "$homedir")" "$(basename "$homedir")"
      chmod 600 "$backup_file"
      chown "$username" "$backup_file"
    fi
  fi
done </etc/passwd