#date: 2022-04-14T16:57:14Z
#url: https://api.github.com/gists/754e76a58b1fd53f52437fb5b8b69212
#owner: https://api.github.com/users/dlydiard

# auto-sign is required if using GitHub Desktop

# Install gpg2
brew install gnupg2

# create a new pgp key (interactive)
gpg --gen-key

# list current keys
gpg --list-secret-keys --keyid-format LONG

# see your gpg public key
gpg --armor --export SEC_KEY_ID

# set gpg key for git
git config --global user.signingkey SEC_KEY_ID

# auto-sign all commits 
git config --global commit.gpgsign true