#date: 2024-05-23T16:52:24Z
#url: https://api.github.com/gists/e5845bedb01ef3f88610c5b9a9f78cfe
#owner: https://api.github.com/users/wilsonsilva

ghcd() {
  if [ -z "$1" ]; then
    echo "Usage: ghcd <username/repo>"
  else
    wd github
    folder_name=$(echo "$1" | cut -d "/" -f1)
    repo_name=$(echo "$1" | cut -d "/" -f2)
    cd "$folder_name/$repo_name"
  fi
}

# requires wd and a wd alias called github. Alternatively, replace wd github above with the absolute path of your Github repos