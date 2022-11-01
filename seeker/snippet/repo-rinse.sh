#date: 2022-11-01T17:05:52Z
#url: https://api.github.com/gists/bbab97a04a29b9ae159d9c393c35e2d7
#owner: https://api.github.com/users/BrandonPacewic

git clean -xfd
git submodule foreach --recursive git clean -xfd
git reset --hard
git submodule foreach --recursive git reset --hard
git submodule update --init --recursive
