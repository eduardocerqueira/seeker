#date: 2024-08-14T18:25:13Z
#url: https://api.github.com/gists/65d429bf9eec149e030ad7d53317f5a1
#owner: https://api.github.com/users/z8leo

#!/bin/bash

# Automate git add, commit and push with a single command with a simple alias within git

git config --global alias.acp '!f() { set -e; git add -A && git commit -m "${1:-changes}" && git push; }; f'

# Example usage: 
# git acp "custom commit message"
# git acp   # Applies "changes" as defualt commit message

# Alias for add and commit
# In some cases, i do not want to push. The alias for this case is:
git config --global alias.ac '!f() { set -e; git add -A && git commit -m "${1:-changes}"; }; f'

# Example usage: 
# git ac "custom commit message"
# git ac   # Applies "changes" as defualt commit message