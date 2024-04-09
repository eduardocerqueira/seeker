#date: 2024-04-09T16:50:18Z
#url: https://api.github.com/gists/d7286deacfd7c8206d1f7f79a361ff78
#owner: https://api.github.com/users/talkingmoose

#!/bin/zsh

# get public IP address
publicIP=$( /usr/bin/curl http://ifconfig.me/ip \
--location \
--silent \
--max-time 10 )

# get GeoIP data
locationData=$( /usr/bin/curl http://ip-api.com/xml/$publicIP \
--location \
--silent \
--max-time 10  )

locationPieces=( country countryCode region regionName city zip lat lon timezone isp org as )

for anItem in $locationPieces
do
	export $anItem="$( /usr/bin/xmllint --xpath "/query/$anItem/text()" <<< "$locationData" - )"
done

echo "<result>$country
$countryCode
$region
$regionName
$city
$zip
$lat
$lon
$timezone
$isp
$org
$as</result>"

exit 0