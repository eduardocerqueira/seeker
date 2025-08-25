#date: 2025-08-25T17:08:27Z
#url: https://api.github.com/gists/246a420c00fbb0b5839f69ac364ef21a
#owner: https://api.github.com/users/williamqwu

curl -sL --compressed "https://recsports.osu.edu/fms/OpenRec?names=Open+Rec+Badminton" | perl -0777 -ne 'while (m{<div class="c-week__day">.*?<h3 class="c-week__header[^>]*>(.*?)</h3>.*?<span class="c-week__event-time">(.*?)</span>}sg){($d,$t)=($1,$2);$d=~s/<[^>]+>//g;$d=~s/^\s+|\s+$//g;$t=~s/<[^>]+>//g;$t=~s/^\s+|\s+$//g;%M=(JANUARY=>"Jan.",FEBRUARY=>"Feb.",MARCH=>"Mar.",APRIL=>"Apr.",MAY=>"May",JUNE=>"Jun.",JULY=>"Jul.",AUGUST=>"Aug.",SEPTEMBER=>"Sep.",OCTOBER=>"Oct.",NOVEMBER=>"Nov.",DECEMBER=>"Dec.");$d=~s/\b([A-Za-z]+)\b/$M{uc $1}||$1/ge;$d=~s/([A-Za-z.]+)\s+(\d{1,2})/$1$2/;print "$d: $t\n"}'