#date: 2021-10-08T17:02:41Z
#url: https://api.github.com/gists/1ebb6898b05d4fdfba8641a06187dd7c
#owner: https://api.github.com/users/mpoquet

#!/usr/bin/env bash

# commands are expected to be run from NUR-Kapack (https://github.com/oar-team/nur-kapack)

# build all packages
nix-build . -A simgrid-327 -o result-sg-327
nix-build . -A simgrid-328 -o result-sg-328
nix-build . -A simgrid-329 -o result-sg-329
nix-build . -A simgrid-327light -o result-sg-327light
nix-build . -A simgrid-328light -o result-sg-328light
nix-build . -A simgrid-329light -o result-sg-329light

# generate closure size for each package
nix-store -qR result-sg-327 | sed -E 's/(.*)/du -bs \1/' | bash | sed -E 's/[[:space:]]+/ /g' | sed -E 'sW/nix/store([^-]*)-(.*)W\2W' > sg327.dat
nix-store -qR result-sg-328 | sed -E 's/(.*)/du -bs \1/' | bash | sed -E 's/[[:space:]]+/ /g' | sed -E 'sW/nix/store([^-]*)-(.*)W\2W' > sg328.dat
nix-store -qR result-sg-329 | sed -E 's/(.*)/du -bs \1/' | bash | sed -E 's/[[:space:]]+/ /g' | sed -E 'sW/nix/store([^-]*)-(.*)W\2W' > sg329.dat
nix-store -qR result-sg-327light | sed -E 's/(.*)/du -bs \1/' | bash | sed -E 's/[[:space:]]+/ /g' | sed -E 'sW/nix/store([^-]*)-(.*)W\2W' > sg327light.dat
nix-store -qR result-sg-328light | sed -E 's/(.*)/du -bs \1/' | bash | sed -E 's/[[:space:]]+/ /g' | sed -E 'sW/nix/store([^-]*)-(.*)W\2W' > sg328light.dat
nix-store -qR result-sg-329light | sed -E 's/(.*)/du -bs \1/' | bash | sed -E 's/[[:space:]]+/ /g' | sed -E 'sW/nix/store([^-]*)-(.*)W\2W' > sg329light.dat