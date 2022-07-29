#date: 2022-07-29T17:12:01Z
#url: https://api.github.com/gists/ac2d6234f9c91957b7f728fbc8dd5f2f
#owner: https://api.github.com/users/skybleu

# From native Terminal
# To verify you are running natively:
arch
-> arm64

# Install node
nvm install stable

# Verify node installation
nvm use stable
node -e 'console.log(process.arch)'
-> arm64