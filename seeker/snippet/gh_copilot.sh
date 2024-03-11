#date: 2024-03-11T16:57:01Z
#url: https://api.github.com/gists/0639f468e4ce5476f9fdaaed12df7bb7
#owner: https://api.github.com/users/lawwu

# https://docs.github.com/en/copilot/github-copilot-in-the-cli/using-github-copilot-in-the-cli

brew install gh
gh auth login
gh extension install github/gh-copilot
gh extension upgrade gh-copilot

# explain command
gh copilot explain

# explain command directly
gh copilot explain "sudo apt-get"

# suggest a command
gh copilot suggest

# suggest a command directly
gh copilot suggest "Install git"
