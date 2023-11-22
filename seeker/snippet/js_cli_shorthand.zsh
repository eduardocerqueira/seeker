#date: 2023-11-22T17:08:42Z
#url: https://api.github.com/gists/d119647ee3dda0be6a54f9cb56e71bfb
#owner: https://api.github.com/users/jbouhier

# Automatically uses the package manager of current directory
# Just type "start" or any other scripts fo your package.json file 

p() {
  if [[ -f bun.lockb ]]; then
    command bun "$@"
  elif [[ -f pnpm-lock.yaml ]]; then
    command pnpm "$@"
  elif [[ -f yarn.lock ]]; then
    command yarn "$@"
  elif [[ -f package-lock.json ]]; then
    command npm "$@"
  else
    command pnpm "$@"
  fi
}