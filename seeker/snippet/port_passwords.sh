#date: 2023-01-05T16:46:58Z
#url: https://api.github.com/gists/f3f592206da51f0cf88fcac1f0babd9b
#owner: https://api.github.com/users/calh

#!/bin/bash
# This script will port old mysql 5.5 passwords to 
# something remotely workable with mysql 8.x.
#
# If we find OLD mysql 4.x password hashes, I'll
# ask to find and set in a new password format.  If
# the password you entered doesn't match the old hash, 
# it repeats and lets you retry a different password.
#
# Run me with something like 
#     ./port_passwords.sh > users.sql
# Then copy users.sql to the new mysql 8.x server and:
#     mysql -uroot < users.sql
#     mysql -uroot -e 'FLUSH PRIVILEGES'

mysql_opts="-uroot --skip-column-names"

# give me the user & host, and I'll prompt to enter
# in a new password, check it against the old mysql 4.x password,
# then return the password hash in the new format
function get_new_pwd()
{
        user=$1
        host=$2
        >&2 echo "User '$user'@'$host' has an OLD mysql 4.x password!"
        while [[ 1 ]]; do
                >&2 echo -n "**********"Find the password and enter the plaintext: "**********"
                read new_pass
                ret= "**********"='$1' and host='$2'" mysql)
                #>&2 echo "ret is '$ret'"
                if [[ "$ret" == "0" ]]; then
                        #echo "$new_pass"
                        mysql $mysql_opts -e "select password('$new_pass')" mysql
                        return 1
                fi
                >&2 echo "That doesn't look like the correct password, try again!"
        done
}

export IFS=$' \n'
for row in $(mysql $mysql_opts -e 'select User,Password,Host from user' mysql); do
        user=$(echo "$row" | awk -F\\t '{print $1}')
        pass=$(echo "$row" | awk -F\\t '{print $2}')
        host=$(echo "$row" | awk -F\\t '{print $3}')
        # skip root user or blank usernames
        if [[ "$user" == "root" || "$user" == "" ]]; then
                continue
        fi
        # port mysql 4.x password hashes
        if [[ $(echo "$pass" | wc -c) -lt 42 ]]; then
                pass=$( get_new_pwd $user $host )
        fi
        #>&2 echo "user: '$user'  pass: '$pass'  host: '$host'"
        echo "CREATE USER '$user'@'$host' IDENTIFIED WITH mysql_native_password AS '$pass';"
        # for each user, select and emit the grants.  Remove the old style grant with password
        mysql $mysql_opts -e "show grants for '$user'@'$host'" | sed 's|$|;|' | sed "s|IDENTIFIED BY PASSWORD '.*'||"
done

>&2 echo
>&2 echo "Done. Don't forget to FLUSH PRIVILEGES after you load this file."
>&2 echoget to FLUSH PRIVILEGES after you load this file."
>&2 echo