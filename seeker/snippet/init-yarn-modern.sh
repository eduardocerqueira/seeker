#date: 2024-04-05T17:00:59Z
#url: https://api.github.com/gists/9829049e95e5827d0e18de5e7c1ec306
#owner: https://api.github.com/users/jamescallumyoung

# exit if any command fials
set -e

# use the latest version of node
nvm use --lts

# save the current (latest) node version so nvm can select it in the future
# (select is with `nvm use`)
node -v > .nvmrc

# corepack is a "package manager"-manager shipped with modern versions of node
# using corepack, we can skip installing yarn/pnpm/etc. and let corepack handle it

# enable corepack
corepack enable

# download the classic version of yarn (if corepack doesn't already have it)
# this version is saved globally, so you can run `yarn init -2` to initialize a new project
# (the init command for Yarn Modern projects is provided by Yarn Classic)
corepack install -g yarn@1

# initialize a modern yarn project
yarn init -2

# download the latest version of yarn modern (if corepack doesn't already have it), and save it in the project's package.json
# (this version will be symlinked so, when running `yarn` inside this project, the latest version is called)
corepack use yarn
corepack up
