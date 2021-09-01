#date: 2021-09-01T13:14:33Z
#url: https://api.github.com/gists/b1cbd3d71f3789e20a6a318d1167cb3d
#owner: https://api.github.com/users/TeiV2

# include in ~/.bashrc or ~/.profile
# allows open programs in terminal without output or any relation
# with the terminal.
start () {
    nohup $@ &>/dev/null & disown
}