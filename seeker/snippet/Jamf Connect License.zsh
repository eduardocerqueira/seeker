#date: 2022-03-30T17:11:22Z
#url: https://api.github.com/gists/07ae935169e863b377c2290c40a660e0
#owner: https://api.github.com/users/talkingmoose

#!/bin/zsh

# Try to read the com.jamf.connect.login.plist file for license information

licenseInfo=$( /usr/bin/defaults read /Library/Managed\ Preferences/com.jamf.connect.login LicenseFile 2> /dev/null | /usr/bin/base64 --decode )

# If the file doesn't exist, try to read the com.jamf.connect.plist file for license information

if [ "$licenseInfo" = "" ]; then
	licenseInfo=$( /usr/bin/defaults read /Library/Managed\ Preferences/com.jamf.connect LicenseFile 2> /dev/null | /usr/bin/base64 --decode )
fi

# Return results to Jamf Pro

if [ "$licenseInfo" = "" ]; then

	# No license file exists for Jamf Connect
	
	echo "<result>No license</result>"
	
else

	# read licensing information from found license file
		
	licensedTo=$( /usr/bin/xpath -e '//key[text()="Name"]/following-sibling::string[1]/text()' 2> /dev/null ) <<< "$licenseInfo"
	licenseKey=$( /usr/bin/xpath -e '//key[text()="LicenseKey"]/following-sibling::string[1]/text()' 2> /dev/null ) <<< "$licenseInfo"
	dateIssued=$( /usr/bin/xpath -e '//key[text()="DateIssued"]/following-sibling::string[1]/text()' 2> /dev/null | /usr/bin/awk '{ print $1 }' ) <<< "$licenseInfo"
	ExpiresOn=$( /usr/bin/xpath -e '//key[text()="ExpirationDate"]/following-sibling::string[1]/text()' 2> /dev/null | /usr/bin/awk '{ print $1 }' ) <<< "$licenseInfo"
	numberOfSeats=$( /usr/bin/xpath -e '//key[text()="NumberOfClients"]/following-sibling::integer[1]/text()' 2> /dev/null ) <<< "$licenseInfo"
	
	# Return license information to Jamf Pro
	
	echo "<result>Licensed to: $licensedTo
License Key: $licenseKey
Date Issued: $dateIssued
Expires On: $ExpiresOn
Number of Seats: $numberOfSeats</result>"

fi

exit 0