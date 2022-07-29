#date: 2022-07-29T17:12:01Z
#url: https://api.github.com/gists/ac2d6234f9c91957b7f728fbc8dd5f2f
#owner: https://api.github.com/users/skybleu

# From Rosetta Terminal (tick checkbox in Terminal right click menu -> Get Info -> Open using Rosetta)
# To verify you are running in Rosetta run:
arch
-> i386

# Install node
nvm install lts/fermium

# Verify node installation
nvm use lts/fermium
node -e 'console.log(process.arch)'
-> x64