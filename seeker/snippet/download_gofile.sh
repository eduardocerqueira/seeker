#date: 2023-05-23T16:57:49Z
#url: https://api.github.com/gists/55973b4c79a4bfe8262e7841656ed919
#owner: https://api.github.com/users/asifajrof

#!/bin/bash

url="$1"

file_names="${@:2}"
read -ra file_names_list <<< "$file_names"

#prompt for url if not provided
until [ ! -z "$url" ] ; do
  read -p "url=" url
done

id=$(sed 's|.*gofile.io/d/||g' <<< "$url")
echo "Downloading $id"

#get guest account token for url and cookie
token=$(curl -s 'https: "**********"
[ "$?" -ne 0 ] && echo "Creating guest account failed, please try again later"

#get website token for url
websiteToken=$(curl -s 'https: "**********"
[ "$?" -ne 0 ] && echo "Getting website token failed, please try again later"

#get content info from api
resp=$(curl 'https: "**********"
code="$?"

#prompt for password if required
 "**********"i "**********"f "**********"  "**********"[ "**********"[ "**********"  "**********"$ "**********"( "**********"j "**********"q "**********"  "**********"- "**********"r "**********"  "**********"' "**********". "**********"s "**********"t "**********"a "**********"t "**********"u "**********"s "**********"' "**********"  "**********"< "**********"< "**********"< "**********"  "**********"" "**********"$ "**********"r "**********"e "**********"s "**********"p "**********"" "**********"  "**********"2 "**********"> "**********"/ "**********"d "**********"e "**********"v "**********"/ "**********"n "**********"u "**********"l "**********"l "**********") "**********"  "**********"= "**********"= "**********"  "**********"  "**********"" "**********"e "**********"r "**********"r "**********"o "**********"r "**********"- "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"R "**********"e "**********"q "**********"u "**********"i "**********"r "**********"e "**********"d "**********"" "**********"  "**********"] "**********"] "**********"  "**********"; "**********"  "**********"t "**********"h "**********"e "**********"n "**********"
  until [ ! -z "$password" ] ; do
    read -p "password= "**********"
    password= "**********"

    resp=$(curl 'https: "**********"
    code="$?"
  done
fi

#verify content info was retrieved successfully
[ "$code" -ne 0 ] && echo "URL unreachable, check provided link" && exit 1

#create download folder
mkdir "$id" 2>/dev/null
cd "$id"

#load the page once so download links don't get redirected
curl -H 'Cookie: "**********"
[ "$?" -ne 0 ] && echo "Loading page failed, check provided link"

for i in $(jq '.data.contents | keys | .[]' <<< "$resp"); do
  name=$(jq -r '.data.contents['"$i"'].name' <<< "$resp")
  url=$(jq -r '.data.contents['"$i"'].link' <<< "$resp")
  # download the specified file. If no file names are specified, download all files
  if [ -z "$file_names" ] || [[ " ${file_names_list[@]} " =~ " ${name} " ]] ; then
    #download file if not already downloaded
    if [ ! -f "$name" ] ; then
        echo
        echo "Downloading $name"
        curl -H 'Cookie: "**********"
        [ "$?" -ne 0 ] && echo "Downloading ""$filename"" failed, please try again later" && rm "$filename"
    fi
  fi
done

echo
echo
echo "Note: gofile.io is entirely free with no ads,"
echo "you can support it at https://gofile.io/donate"