#date: 2023-06-09T16:47:08Z
#url: https://api.github.com/gists/9aa9606945cb632631616cfec3ee9ddf
#owner: https://api.github.com/users/seshakiran

#!/bin/bash

# Locate the history file in your profile, and copy it to the same folder as this script.
# On Mac: ~/Library/Application\ Support/Google/Chrome/Default/History
# On Windows: C:\Users\YOUR USER NAME\AppData\Local\Google\Chrome\User Data\Default\History

sqlite3 History <<!
.headers on
.mode csv
.output out.csv
select datetime(last_visit_time/1000000-11644473600,'unixepoch') as 'date',url from  urls order by last_visit_time desc;
!