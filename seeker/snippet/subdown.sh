#date: 2024-01-10T17:09:54Z
#url: https://api.github.com/gists/eacf6e4cbdc4ff5206f0f4b4457029f7
#owner: https://api.github.com/users/ramosslyz

#!/bin/bash
NC="\e[0m"
COL="\e[31m"
silent=False

while [ -n "$1" ]; do
  case $1 in
        -s|--silent)
            silent='true'
            ;;
  esac
  shift
done

shred -u -f file.ods dutchsub.txt file.csv 2> /dev/null &> /dev/null
url=$(echo -e "https://www.communicatierijk.nl$(curl -ks https://www.communicatierijk.nl/vakkennis/r/rijkswebsites/verplichte-richtlijnen/websiteregister-rijksoverheid | grep ".ods" | sed -n 's/.*href="\([^"]*\).*/\1/p')")

[ "$silent" == "False" ] && echo -e "FILE PATH:$COL $url $NC"
wget -O file.ods $url &> /dev/null
libreoffice5.3 --convert-to csv file.ods --headless 2> /dev/null &> /dev/null
cat file.csv | tr ',' ' ' | awk '{print $1}' | sed 's/http:\/\/\|https:\/\///g' | sed -e '1,3d' | sed 's/\.com.*/.com/' | sed 's/\.nl.*/.nl/' | sed 's/\.org.*/.org/' | sed 's/\.eu.*/.eu/' | awk -F\. '{print $(NF-1) FS $NF}' | sed -E '/www\.org|www\.nl|www\.com|www\.eu|org\.org|eu\.eu|nl\.nl|eu\.eu/d' | sort -u | uniq -u | anew -q dutchsub.txt
[ "$silent" == "False" ] && echo -e "Total subdomains found:$COL $(cat dutchsub.txt | wc -l) $NC\n"
sleep 2s
cat dutchsub.txt
[ "$silent" == "False" ] && echo ""
[ "$silent" == "False" ] && echo -e "Total subdomains found:$COL $(cat dutchsub.txt | wc -l) $NC"