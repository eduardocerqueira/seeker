#date: 2023-07-03T17:00:37Z
#url: https://api.github.com/gists/60e7aca9351ce8d44d73ffbda003d014
#owner: https://api.github.com/users/wizardishungry

function gt { go run $(go list -f '{{join .Imports "\n" }}' -tags tools tools.go | egrep "\/$1\$" ) }