#date: 2024-11-13T16:53:41Z
#url: https://api.github.com/gists/c435bafcd79beb91b65e354fc907efe8
#owner: https://api.github.com/users/wteuber

# https://gist.github.com/wteuber/c435bafcd79beb91b65e354fc907efe8
# macOS/zsh

# Apply Ruby's keyword argument and hash value omission in a git repo

# First occurence only
git grep -lP '\b([A-Za-z0-9_]+): \1\b[,\)]' -- '*.rb' | head -1 | xargs -I {} sed -i -E 's/\b([A-Za-z0-9_]+): \1\b([,\)])/\1:\2/g' "{}"

# All occurences
git grep -lP '\b([A-Za-z0-9_]+): \1\b[,\)]' -- '*.rb' | xargs -I {} sed -i -E 's/\b([A-Za-z0-9_]+): \1\b([,\)])/\1:\2/g' "{}"

