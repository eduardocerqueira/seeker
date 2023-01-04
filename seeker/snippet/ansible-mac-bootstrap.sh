#date: 2023-01-04T17:10:27Z
#url: https://api.github.com/gists/af1afff648055003273e1170099dba2e
#owner: https://api.github.com/users/dansonnenburg

# Mac Setup
# Install Brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Install wget
brew install wget
# Configure ansible
wget -qO --no-check-certificate - https://gist.githubusercontent.com/dansonnenburg/ad3c10b4674e86360d144c4522ac0979/raw/a83a3e3f0178db16da1c61f53e1244db56b966b0/ansible-mac-endpoint.sh | sudo bash