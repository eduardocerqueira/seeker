#date: 2022-10-13T17:25:38Z
#url: https://api.github.com/gists/28159835ca9b6d2ca0aa42010d5f7690
#owner: https://api.github.com/users/aequitz

git clean -ffxd -e .venv
git submodule foreach --recursive git clean -ffxd
git reset --hard
git submodule foreach --recursive git reset --hard
git submodule update --init --recursive