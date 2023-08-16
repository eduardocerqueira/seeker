#date: 2023-08-16T16:39:33Z
#url: https://api.github.com/gists/ca112c164ad8598f48fb5aebe9f5a0b9
#owner: https://api.github.com/users/WhatsARanjit

if [ -f "dontdoit" ]; then
  RESULT="true"
else 
  RESULT="false"
fi
echo "{ \"check\": \"$RESULT\" }"