#date: 2023-06-16T16:51:47Z
#url: https://api.github.com/gists/a2a7659bdf4ba1261e9c6688ba9c55e7
#owner: https://api.github.com/users/DanEdens

#!/bin/bash

repo_url="https://github.com/DanEdens/resume.git"
repo_folder="resume"

if [ ! -d "$repo_folder" ]; then
    echo "Cloning repository..."
    git clone "$repo_url" "$repo_folder"
    cd "$repo_folder"
else
    echo "Repository folder already exists. Performing commit and push."
    cd "$repo_folder"
    git add --all
    git commit -m "Commit message"
    git push origin master
    cd ..
    rm -rf "$repo_folder"
fi
