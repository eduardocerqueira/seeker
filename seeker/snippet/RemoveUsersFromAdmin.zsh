#date: 2022-04-28T17:07:01Z
#url: https://api.github.com/gists/e0536816d0982a0a6f33afe3c652c69f
#owner: https://api.github.com/users/talkingmoose

#!/bin/zsh

# these local accounts will not be removed from admins
# one account name per line; keep the beginning and closing quotes

exceptionsList="talkingmoose
bill.smith
oszein
jamfadmin"

# list all users with UIDs greater than or equal to 500

localUsers=$( /usr/bin/dscl /Local/Default -list /Users uid | /usr/bin/awk '$2 >= 500 { print $1 }' )
echo "List of local accounts:
$localUsers\n"

# remove all but "ARD" and "IT" users from local admins group

while IFS= read aUser
do
	if [ ! $( /usr/bin/grep "$aUser" <<< "$exceptionsList" ) ] ; then
		/usr/sbin/dseditgroup -o edit -d "$aUser" -t user admin
		echo "Removed user: $aUser from admins group"
	fi
done <<< "$localUsers"

exit 0