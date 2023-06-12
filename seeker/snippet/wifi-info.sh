#date: 2023-06-12T17:05:47Z
#url: https://api.github.com/gists/c9ec30e0824e85f6235ba11177a3bee9
#owner: https://api.github.com/users/verityj

#!/bin/zsh

# curl -s hides the progress bar (verbose is -v)
# curl --progress-bar shows a simplier progress meter
# curl -S with silence activated still shows errors (or --show-error)

 "**********"# "**********"  "**********"A "**********"d "**********"m "**********"i "**********"n "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"b "**********"e "**********"t "**********"w "**********"e "**********"e "**********"n "**********"  "**********"q "**********"u "**********"o "**********"t "**********"e "**********"s "**********": "**********"
admin= "**********"

# First command:
# curl -X POST http: "**********": "admin", "password": "admin-password-here"}'
# Second command:
# curl http: "**********":Bearer your-token-here"
# (Script from https://github.com/christopherjnelson/Arcadyan-5G-Web-Admin/issues/4)

curl -sSX POST http: "**********": "admin", "password": "'$admin'"}' > temp.txt
token="${$(sed '6!d' temp.txt): "**********":284}"
rm temp.txt

curl -sS http: "**********":Bearer $token"
echo -e # move to next line to complete output
