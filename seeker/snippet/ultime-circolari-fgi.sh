#date: 2025-03-27T17:04:23Z
#url: https://api.github.com/gists/8bdf95e07dd87e77f8ca6d9420386684
#owner: https://api.github.com/users/checco

curl --silent https://www.federginnastica.it/documenti/circolari.html | grep "pd-button-download" | awk '{print $6}' | sed "s/:/ /g" | awk '{print $2}' | tr -d '"'