#date: 2023-10-17T17:03:43Z
#url: https://api.github.com/gists/253dcf2521a35853fdedad74b6f49572
#owner: https://api.github.com/users/assholehoff

brew install gpg

# Generating and exporting
gpg --gen-key
gpg --armor --export <email> > ~/.backups/gpg_git_public.key
gpg --armor --export-secret-key <email> > ~/.backups/gpg_git_private.key
chmod 600 ~/.backups/gpg_git_public.key
chomd 600 ~/.backups/gpg_git_private.key

# Importing
gpg --armor --import ~/.backups/gpg_git_private.key

gpg --list-secret-keys
# Copy the 8-digit id of the key (in the `sec` section)
git config --global user.signingkey <8-digit key id>
git config --global commit.gpgsign true
vim ~/.gnupg/key.txt
# Put the passpharse for the GPG key to the first line of the file
chmod 600 ~/.gnupg/key.txt
vim /usr/local/bin/gpg-with-key
# Add lines from the gpg-with-key file of this gist
chmod 755 /usr/local/bin/gpg-with-key
git config --global gpg.program "/usr/local/bin/gpg-with-key"