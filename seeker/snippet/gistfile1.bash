#date: 2023-05-16T17:07:18Z
#url: https://api.github.com/gists/9946b9577697d47785ce91f200ce0f3e
#owner: https://api.github.com/users/MszBednarski

# Current directory LOC analytics
brew install cloc
echo "alias lines='cloc $(git ls-files)'" >> ~/.zshrc
source ~/.zshrc
# cd directory you like and just:
lines