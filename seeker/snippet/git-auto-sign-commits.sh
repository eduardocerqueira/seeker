#date: 2022-11-02T17:16:38Z
#url: https://api.github.com/gists/2b8dbd285038c58d936ba552bd2b4381
#owner: https://api.github.com/users/TomaszWaszczyk

# Generate a new pgp key: (better to use gpg2 instead of gpg in all below commands)
gpg --gen-key
# maybe you need some random work in your OS to generate a key. so run this command: `find ./* /home/username -type d | xargs grep some_random_string > /dev/null`

# check current keys:
gpg --list-secret-keys --keyid-format LONG

# See your gpg public key:
gpg --armor --export YOUR_KEY_ID
# YOUR_KEY_ID is the hash in front of `sec` in previous command. (for example sec 4096R/234FAA343232333 => key id is: 234FAA343232333)

# Set a gpg key for git:
git config --global user.signingkey your_key_id

# To sign a single commit:
git commit -S -a -m "Test a signed commit"

# Auto-sign all commits globaly
git config --global commit.gpgsign true