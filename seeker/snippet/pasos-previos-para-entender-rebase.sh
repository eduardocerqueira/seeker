#date: 2022-02-21T17:08:40Z
#url: https://api.github.com/gists/cd7d99aef5ce9b4b642983cd1bc532d6
#owner: https://api.github.com/users/repositorioinformatico

#!/usr/bin/env bash
git init
touch f1
git add f1
git commit -m "c 1 en master"
touch f2
git add f2
git commit -m "c 2 en master"
git checkout -b feature
git checkout master
touch f3
git add f3
git commit -m "c 3 en master"
git --no-pager log --oneline --graph
git checkout feature
touch change1
git add change1
git commit -m "c 1 en feature"
touch change2
git add change2
git commit -m "c 2 en feature"
git --no-pager log --oneline --graph
#hasta aquí, con esta estructura del repo, ahora podremos entender el rebase y compararlo con el merge