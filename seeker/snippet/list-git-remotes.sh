#date: 2023-06-02T16:53:06Z
#url: https://api.github.com/gists/7147a20b0045da2794f341366d82db34
#owner: https://api.github.com/users/mak3r

#!/bin/sh

for f in $(ls -1); do 
  pushd $f; 
  
  echo -e "\n\n******" >> ~/dev/remotes.txt;
  echo $f >> ~/dev/remotes.txt;
  echo "******" >> ~/dev/remotes.txt;
  git remote show origin >> ~/dev/remotes.txt 2>&1; 
  popd; 
  
done;
