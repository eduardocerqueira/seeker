#date: 2022-02-17T17:11:18Z
#url: https://api.github.com/gists/09bf1c0f0f44ea2bd3ba32bd51aad7d7
#owner: https://api.github.com/users/fraigo

curl -s --user 'api:key-{$APIKEY}' \
 https://api.mailgun.net/v3/{$DOMAIN}/messages \
 -F from='{$NAME} <{$USER}@{$DOMAIN}>' \
 -F to='{$EMAIL}' \
 -F subject="{$SUBJECT}" \
 -F text="{$MESSAGE}" \
 -F attachment=@./path/to/attachment