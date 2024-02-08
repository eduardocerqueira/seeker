#date: 2024-02-08T17:09:21Z
#url: https://api.github.com/gists/696acf8541f3b0ec3ba202a8ddbe4daf
#owner: https://api.github.com/users/ghelobytes

kdecsec(){kubectl get secret $1 -o go-template='{{range $k,$v : "**********"intf "%s => " $k}}{{if not $v}}{{$v}}{{else}}{{$v | base64decode}}{{end}}{{"\n"}}{{end}}';}