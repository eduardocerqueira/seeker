#date: 2023-07-05T17:08:43Z
#url: https://api.github.com/gists/e0bbfa8672eddc7ca2fdfe9c032aa318
#owner: https://api.github.com/users/kazajhodo

db_pull_command:
  command: |
    set -x   # You can enable bash debugging output by uncommenting
    set -eu -o pipefail
    ls /var/www/html/.ddev >/dev/null # This just refreshes stale NFS if possible
    pushd /var/www/html/.ddev/.downloads >/dev/null
    connection=$(terminus connection:info ${project} --field='MySQL Command')
    connection=${connection/'mysql'/'mysqldump -v'}
    eval "$connection --single-transaction --default-character-set=utf8mb4 --quick | gzip > db.sql.gz"