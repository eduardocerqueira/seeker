#date: 2025-04-24T16:49:50Z
#url: https://api.github.com/gists/a3ba7997fcb0a1578d012f39b451b7a6
#owner: https://api.github.com/users/gladiopeace

curl -s https://api.github.com/users/milanboers/repos | grep \"clone_url\" | awk '{print $2}' | sed -e 's/"//g' -e 's/,//g' | xargs -n1 git clone