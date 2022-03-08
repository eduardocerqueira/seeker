#date: 2022-03-08T17:01:12Z
#url: https://api.github.com/gists/743c0e78368c374e1472cfab54420a8b
#owner: https://api.github.com/users/Facuunz5

#!/usr/bin/env bash
git init
touch f1
git add f1
git commit -m "f1"
touch f2
git add f2
git commit -m "f2"
touch f3
git add f3
git commit -m "f3-este-commit-rompe"
touch f4
git add f4
git commit -m "f4-este-commit-rompe"
touch f5
git add f5
git commit -m "f5"
git --no-pager log --oneline --all 
#hasta aqui
git rebase --interactive <commit-antes-de-lo-que-queremos-borrar>
#cambiar 'pick' por 'drop'
