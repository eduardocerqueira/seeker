#date: 2024-09-26T17:09:23Z
#url: https://api.github.com/gists/f971e367780bb50dd8dbf0212af6442f
#owner: https://api.github.com/users/gustafla

#!/bin/bash
set -eu

curl -sS "$SOURCE" | \
curl -sS -u "$AUTH" -k -X PUT -H 'Content-Type: text/calendar' --data-binary @- "$COLLECTION"
curl -sS -u "$AUTH" -k -X PROPPATCH --data-binary @- "$COLLECTION" >/dev/null <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<propertyupdate xmlns="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav" xmlns:CR="urn:ietf:params:xml:ns:carddav" xmlns:CS="http://calendarserver.org/ns/" xmlns:I="http://apple.com/ns/ical/" xmlns:INF="http://inf-it.com/ns/ab/">
  <set>
    <prop>
      <C:supported-calendar-component-set>
        <C:comp name="VEVENT" />
      </C:supported-calendar-component-set>
      <displayname>$DISPLAYNAME</displayname>
      <I:calendar-color>$COLOR</I:calendar-color>
    </prop>
  </set>
</propertyupdate>
EOF